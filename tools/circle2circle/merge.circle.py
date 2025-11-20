#!/usr/bin/env python3

import sys
import os
import argparse
import hashlib
from typing import List, Dict, Tuple, Union, Optional

import circle
import o2o


def get_signature_key(filepath: str) -> str:
    """Extracts the filename without extension to use as a signature key."""
    filename = os.path.basename(filepath)
    if filename.endswith('.circle'):
        return filename[:-7]
    return filename


def create_tensor_map(subgraph: 'circle.SubGraphT',
                      tensor_indices: List[int]) -> List['circle.TensorMapT']:
    """Creates a list of TensorMap for SignatureDef inputs/outputs."""
    tensor_maps = []
    for idx in tensor_indices:
        tensor = subgraph.tensors[idx]
        tensor_map = circle.TensorMapT()
        tensor_map.name = tensor.name
        tensor_map.tensorIndex = idx
        tensor_maps.append(tensor_map)
    return tensor_maps


def merge_models(model_paths: List[str]) -> None:
    if not model_paths:
        return

    # Load all models
    models: List['circle.ModelT'] = []
    for path in model_paths:
        models.append(o2o.load_circle_model(path))

    # Create new merged model
    merged_model = circle.ModelT()
    # Use the max version among all models
    merged_model.version = max(m.version for m in models)
    merged_model.description = f"Merged from {', '.join([os.path.basename(p) for p in model_paths])}"

    merged_model.operatorCodes = []
    merged_model.subgraphs = []
    merged_model.buffers = []
    merged_model.signatureDefs = []

    # 1. Merge Operator Codes (Deduplication)
    # Map (type, code) -> new_index
    opcode_map: Dict[Tuple[str, Union[int, str]], int] = {}
    # List of maps for each model: old_index -> new_index
    model_opcode_maps: List[Dict[int, int]] = []

    for model in models:
        local_map = {}
        for old_idx, op_code in enumerate(model.operatorCodes):
            key = o2o.get_operator_code_key(op_code)
            if key in opcode_map:
                new_idx = opcode_map[key]
            else:
                new_idx = len(merged_model.operatorCodes)
                merged_model.operatorCodes.append(op_code)
                opcode_map[key] = new_idx
            local_map[old_idx] = new_idx
        model_opcode_maps.append(local_map)

    # 2. Merge Buffers (with deduplication)
    # Buffer 0 is always empty sentinel.
    merged_model.buffers.append(circle.BufferT())  # Sentinel

    # Map buffer content hash -> merged buffer index (for deduplication)
    buffer_hash_map: Dict[bytes, int] = {}
    # The sentinel buffer (empty) hash
    buffer_hash_map[hashlib.sha256(bytes()).digest()] = 0

    # List of maps for each model: old_index -> new_index
    model_buffer_maps: List[Dict[int, int]] = []

    for model in models:
        local_map = {}
        # Map model's sentinel (0) to merged sentinel (0)
        local_map[0] = 0

        # Process other buffers with deduplication
        for old_idx in range(1, len(model.buffers)):
            buffer = model.buffers[old_idx]

            # Create a hash digest from buffer data
            if buffer.data is not None:
                buffer_hash = hashlib.sha256(bytes(buffer.data)).digest()
            else:
                buffer_hash = hashlib.sha256(bytes()).digest()

            # Check if this buffer already exists
            if buffer_hash in buffer_hash_map:
                # Reuse existing buffer
                new_idx = buffer_hash_map[buffer_hash]
            else:
                # Add new buffer
                new_idx = len(merged_model.buffers)
                merged_model.buffers.append(buffer)
                buffer_hash_map[buffer_hash] = new_idx

            local_map[old_idx] = new_idx

        model_buffer_maps.append(local_map)

    # 3. Merge Subgraphs
    # We assume 1 subgraph per input model.

    for model_idx, model in enumerate(models):
        if not model.subgraphs:
            # Create empty subgraph if none exists (though unlikely for valid models)
            subgraph = circle.SubGraphT()
            merged_model.subgraphs.append(subgraph)
            subgraph_idx = len(merged_model.subgraphs) - 1
        else:
            # Take the first subgraph
            subgraph = model.subgraphs[0]

            # Update Operator Opcode Indices
            for op in subgraph.operators:
                if op.opcodeIndex in model_opcode_maps[model_idx]:
                    op.opcodeIndex = model_opcode_maps[model_idx][op.opcodeIndex]

            # Update Tensor Buffer Indices
            for tensor in subgraph.tensors:
                if tensor.buffer in model_buffer_maps[model_idx]:
                    tensor.buffer = model_buffer_maps[model_idx][tensor.buffer]

            merged_model.subgraphs.append(subgraph)
            subgraph_idx = len(merged_model.subgraphs) - 1

        # 4. Create SignatureDefs
        sig = circle.SignatureDefT()
        sig.signatureKey = get_signature_key(model_paths[model_idx])
        sig.subgraphIndex = subgraph_idx
        if model.subgraphs:
            sig.inputs = create_tensor_map(subgraph, subgraph.inputs)
            sig.outputs = create_tensor_map(subgraph, subgraph.outputs)
        merged_model.signatureDefs.append(sig)

    # Save to stdout
    o2o.save_model_to_stdout(merged_model)


def main():
    parser = argparse.ArgumentParser(description='Merge multiple circle models into one.')
    parser.add_argument('models', nargs='+', help='Paths to the circle models to merge')
    args = parser.parse_args()

    for model_path in args.models:
        if not os.path.exists(model_path):
            print(f"Error: File not found: {model_path}", file=sys.stderr)
            sys.exit(1)

    merge_models(args.models)


if __name__ == '__main__':
    main()
