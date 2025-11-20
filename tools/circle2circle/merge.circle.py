#!/usr/bin/env python3

import sys
import os
import argparse
from typing import List, Optional, Tuple, Dict, Any
import o2o
import circle

# Import specific Circle types for better type annotations
from circle import (TensorT, OperatorT, SubGraphT, ModelT, BufferT, OperatorCodeT,
                    BuiltinOperator, TensorType, SignatureDefT, TensorMapT)


def merge_operator_codes_with_deduplication(
        model1: 'circle.ModelT',
        model2: 'circle.ModelT') -> Tuple[List['circle.OperatorCodeT'], Dict[int, int]]:
    """Merge operator codes from two models while removing duplicates.

    Args:
        model1: First Circle model
        model2: Second Circle model

    Returns:
        tuple: (merged_operator_codes, model2_to_merged_mapping)
    """
    # Start with first model's operator codes
    merged_operator_codes = list(model1.operatorCodes)

    # Create mapping table for operator codes
    # key: (builtinCode, customCode), value: new_index
    opcode_mapping = {}

    # Register first model's operator codes
    for i, op_code in enumerate(model1.operatorCodes):
        key = o2o.get_operator_code_key(op_code)
        opcode_mapping[key] = i

    # Process second model's operator codes (check for duplicates)
    model2_to_merged_mapping = {}  # model2's index → merged index

    for i, op_code in enumerate(model2.operatorCodes):
        key = o2o.get_operator_code_key(op_code)

        if key in opcode_mapping:
            # Duplicate operator code - use existing index
            model2_to_merged_mapping[i] = opcode_mapping[key]
        else:
            # New operator code - add it
            new_index = len(merged_operator_codes)
            merged_operator_codes.append(op_code)
            opcode_mapping[key] = new_index
            model2_to_merged_mapping[i] = new_index

    return merged_operator_codes, model2_to_merged_mapping


def create_tensor_map_list(subgraph: 'circle.SubGraphT',
                           tensor_indices: List[int]) -> List['circle.TensorMapT']:
    """Convert tensor indices to TensorMap objects for SignatureDef.

    Args:
        subgraph: Subgraph containing the tensors
        tensor_indices: List of tensor indices

    Returns:
        list: List of TensorMapT objects
    """
    tensor_maps = []
    o2o.log(f"Creating tensor maps for {len(tensor_indices)} tensors")

    for i, tensor_idx in enumerate(tensor_indices):
        # Skip optional inputs (-1 indicates unused optional input)
        if tensor_idx == -1:
            continue

        # Ensure tensor index is valid
        if 0 <= tensor_idx < len(subgraph.tensors):
            tensor_map = circle.TensorMapT()

            # Get tensor name, use fallback if no name exists
            tensor_name = o2o.get_tensor_name(subgraph.tensors[tensor_idx])
            if not tensor_name:
                tensor_name = f"tensor_{tensor_idx}"

            # Encode name as UTF-8 bytes for FlatBuffers compatibility
            tensor_map.name = tensor_name.encode('utf-8')
            tensor_map.tensorIndex = int(tensor_idx)  # Convert numpy.int32 to int

            o2o.log(f"  TensorMap {i}: name='{tensor_name}', index={tensor_idx}")
            tensor_maps.append(tensor_map)
        else:
            o2o.log(f"Warning: Invalid tensor index {tensor_idx} in signature creation")

    return tensor_maps


def create_signatures(model: 'circle.ModelT', sig_names: List[str]) -> None:
    """Create signature definitions for the merged model.

    Args:
        model: Merged Circle model
        sig_names: List of signature names for each subgraph (must match subgraph count)
    """
    if not hasattr(model, 'signatureDefs'):
        model.signatureDefs = []

    signatures = []

    for idx, sig_name in enumerate(sig_names):
        sig = circle.SignatureDefT()
        sig.subgraphIndex = idx
        sig.signatureKey = sig_name.encode('utf-8')

        subgraph = model.subgraphs[idx]
        sig.inputs = create_tensor_map_list(subgraph, subgraph.inputs) if list(
            subgraph.inputs) else []
        sig.outputs = create_tensor_map_list(subgraph, subgraph.outputs) if list(
            subgraph.outputs) else []

        signatures.append(sig)

    model.signatureDefs = signatures


def merge_models_with_signatures(model1: 'circle.ModelT', model2: 'circle.ModelT',
                                 sig_name_0: str, sig_name_1: str) -> 'circle.ModelT':
    """Merge two Circle models by keeping subgraphs separate and adding signatures.

    Args:
        model1: First Circle model
        model2: Second Circle model
        sig_name_0: Signature name for first subgraph
        sig_name_1: Signature name for second subgraph

    Returns:
        circle.ModelT: Merged model with signatures
    """
    # Validate that both models have exactly one subgraph
    if not model1.subgraphs or len(model1.subgraphs) != 1:
        o2o.log("Error: First model must have exactly one subgraph")
        sys.exit(1)

    if not model2.subgraphs or len(model2.subgraphs) != 1:
        o2o.log("Error: Second model must have exactly one subgraph")
        sys.exit(1)

    o2o.log(f"Merging models:")
    o2o.log(
        f"  Model 1: {len(model1.subgraphs[0].tensors)} tensors, {len(model1.subgraphs[0].operators)} operators"
    )
    o2o.log(
        f"  Model 2: {len(model2.subgraphs[0].tensors)} tensors, {len(model2.subgraphs[0].operators)} operators"
    )

    # Step 1: Merge buffers (simple append)
    merged_buffers = list(model1.buffers) + list(model2.buffers)
    buffer_offset = len(model1.buffers)

    # Step 2: Merge operator codes with deduplication
    merged_operator_codes, model2_opcode_mapping = merge_operator_codes_with_deduplication(
        model1, model2)

    # Step 3: Create merged subgraphs
    merged_subgraphs = []

    # First subgraph (keep as-is, no index remapping needed)
    subgraph0 = model1.subgraphs[0]
    merged_subgraphs.append(subgraph0)

    # Second subgraph (needs index remapping)
    subgraph1 = model2.subgraphs[0]

    # Remap buffer indices in second subgraph tensors
    for tensor in subgraph1.tensors:
        if tensor.buffer is not None and tensor.buffer != 0:
            tensor.buffer += buffer_offset

    # Remap operator code indices in second subgraph operators
    for operator in subgraph1.operators:
        if operator.opcodeIndex is not None:
            operator.opcodeIndex = model2_opcode_mapping[operator.opcodeIndex]

    merged_subgraphs.append(subgraph1)

    # Step 4: Create final merged model
    merged_model = circle.ModelT()
    merged_model.buffers = merged_buffers
    merged_model.operatorCodes = merged_operator_codes
    merged_model.subgraphs = merged_subgraphs

    # Step 5: Create signatures
    create_signatures(merged_model, [sig_name_0, sig_name_1])

    o2o.log(f"Merge completed:")
    o2o.log(f"  Total buffers: {len(merged_buffers)}")
    o2o.log(f"  Total operator codes: {len(merged_operator_codes)}")
    o2o.log(f"  Total subgraphs: {len(merged_subgraphs)}")
    o2o.log(f"  Signatures: ['{sig_name_0}', '{sig_name_1}']")

    return merged_model


def main():
    """Main function to merge two Circle models with signatures."""
    # This script merges multiple Circle model files into a single model.
    # It keeps each input model as a separate subgraph and adds a signature
    # for each subgraph. If signature names are not provided via --sig-names,
    # they are derived from the input filenames (without the .circle extension).
    parser = argparse.ArgumentParser(
        description='Merge multiple Circle models (as subgraphs) with signatures')
    # One or more Circle model files to merge, e.g. in1.circle in2.circle ...
    parser.add_argument(
        'circles',
        nargs='+',
        help='Circle model files to merge (e.g., in1.circle in2.circle ...)')
    # Optional signature names for each subgraph, separated by semicolons.
    # Must match the number of input files. If omitted, names are taken from the
    # input filenames (without the .circle extension).
    parser.add_argument(
        '--sig-names',
        default=None,
        help=
        'Signature names for subgraphs (semicolon‑separated). If omitted, derived from input filenames.'
    )
    args = parser.parse_args()

    # Currently only support 2 models
    if len(args.circles) != 2:
        o2o.log("Error: Currently only 2 Circle models are supported")
        sys.exit(1)

    # Parse signature names
    if args.sig_names is None:
        # Use filenames without .circle extension as signature names
        sig_names = [os.path.splitext(os.path.basename(f))[0] for f in args.circles]
    else:
        # Use user-provided signature names
        sig_names = args.sig_names.split(';')
        if len(sig_names) != len(args.circles):
            o2o.log(
                f"Error: --sig-names must contain exactly {len(args.circles)} names separated by semicolon"
            )
            sys.exit(1)
        sig_names = [name.strip() for name in sig_names]

    # Validate signature names are not empty
    for i, sig_name in enumerate(sig_names):
        if not sig_name:
            o2o.log(f"Error: Signature name {i+1} cannot be empty")
            sys.exit(1)

    sig_name_0, sig_name_1 = sig_names[0], sig_names[1]

    o2o.log(f"Loading models...")
    o2o.log(f"  First model: {args.circles[0]}")
    o2o.log(f"  Second model: {args.circles[1]}")
    o2o.log(f"  Signature names: ['{sig_name_0}', '{sig_name_1}']")

    # Load both models explicitly
    try:
        model0 = o2o.load_circle_model(args.circles[0])
        model1 = o2o.load_circle_model(args.circles[1])
    except Exception as e:
        o2o.log(f"Error loading models: {e}")
        sys.exit(1)

    # Merge models with signatures
    try:
        merged_model = merge_models_with_signatures(model0, model1, sig_name_0,
                                                    sig_name_1)
    except Exception as e:
        o2o.log(f"Error merging models: {e}")
        sys.exit(1)

    # Output to stdout
    try:
        o2o.save_model_to_stdout(merged_model)
        o2o.log("Successfully saved merged model to stdout")
    except Exception as e:
        o2o.log(f"Error saving merged model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
