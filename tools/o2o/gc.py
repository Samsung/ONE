#!/usr/bin/env python3

import sys
import circle
import o2o
from typing import List, Optional

# Import specific Circle types for better type annotations
from circle import (TensorT, OperatorT, SubGraphT, ModelT, BufferT, OperatorCodeT,
                    BuiltinOperator, TensorType)


def get_tensor_name(tensor: TensorT) -> Optional[str]:
    """Get tensor name as string, handling bytes conversion"""
    if tensor.name:
        return tensor.name.decode('utf-8') if isinstance(tensor.name,
                                                         bytes) else tensor.name
    return None


def find_unused_tensors_in_subgraph(subgraph: SubGraphT) -> List[int]:
    """
    Finds and returns the indices of unused tensors in a given subgraph.
    This function uses the Object API for mutable subgraph objects.

    Args:
        subgraph: The Circle mutable subgraph object (SubGraphT).

    Returns:
        list: A list of integer indices representing unused tensors.
    """
    num_tensors = len(subgraph.tensors)
    if num_tensors == 0:
        return []

    used_tensor_indices = set()
    output_tensor_indices = set()

    # Collect output tensor indices
    if subgraph.outputs is not None and len(subgraph.outputs) > 0:
        for out in subgraph.outputs:
            output_tensor_indices.add(out)

    # Collect input tensor indices from all operators
    for operator in subgraph.operators:
        if operator.inputs is not None and len(operator.inputs) > 0:
            for inp in operator.inputs:
                if inp != -1:
                    used_tensor_indices.add(inp)

    # A tensor is unused if it's not used by any operator AND not an output of the subgraph
    unused_indices = []
    for i in range(num_tensors):
        if i not in used_tensor_indices and i not in output_tensor_indices:
            unused_indices.append(i)

    return unused_indices


def find_unused_buffers(model) -> List[int]:
    """
    Finds and returns the indices of unused buffers in the model.
    This function works with both Native API (read-only) and Object API (mutable) model objects.

    Args:
        model: The Circle model object (read-only or mutable).

    Returns:
        list: A list of integer indices representing unused buffers.
    """
    # Handle both Native API and Object API
    if hasattr(model, 'BuffersLength'):
        # Native API
        if not model.BuffersLength():
            return []

        used_buffer_indices = set()

        # Collect buffer indices from all tensors in all subgraphs
        for i in range(model.SubgraphsLength()):
            subgraph = model.Subgraphs(i)
            if subgraph:
                for j in range(subgraph.TensorsLength()):
                    tensor = subgraph.Tensors(j)
                    if tensor and tensor.Buffer() != -1:  # -1 indicates no buffer
                        used_buffer_indices.add(tensor.Buffer())

        # A buffer is unused if it's not referenced by any tensor
        unused_indices = []
        for i in range(model.BuffersLength()):
            if i not in used_buffer_indices:
                unused_indices.append(i)

        return unused_indices
    else:
        # Object API
        if not model.buffers:
            return []

        used_buffer_indices = set()

        # Collect buffer indices from all tensors in all subgraphs
        for subgraph in model.subgraphs:
            for tensor in subgraph.tensors:
                if tensor.buffer != -1:  # -1 indicates no buffer
                    used_buffer_indices.add(tensor.buffer)

        # A buffer is unused if it's not referenced by any tensor
        unused_indices = []
        for i in range(len(model.buffers)):
            if i not in used_buffer_indices:
                unused_indices.append(i)

        return unused_indices


def remove_tensors_and_update_model(model: ModelT, subgraph_index_to_modify: int,
                                    tensor_indices_to_remove: List[int]) -> List[int]:
    """
    Removes specified tensors from the model and updates all relevant references.
    This function uses the Object API for mutable model/subgraph/operator objects.

    Args:
        model: The mutable Circle model object (ModelT).
        subgraph_index_to_modify (int): The index of the subgraph to modify.
        tensor_indices_to_remove (list): A list of tensor indices to remove.
                                         Must be sorted in descending order.

    Returns:
        list: The list of tensor indices that were actually removed.
    """
    if not model.subgraphs or subgraph_index_to_modify >= len(model.subgraphs):
        o2o.log(
            f"Error: Invalid subgraph index {subgraph_index_to_modify} for modification.")
        return []

    subgraph = model.subgraphs[subgraph_index_to_modify]
    removed_indices = []

    # Sort in descending order to avoid index shifting issues during removal
    for tensor_idx in sorted(tensor_indices_to_remove, reverse=True):
        if 0 <= tensor_idx < len(subgraph.tensors):
            tensor_name = get_tensor_name(subgraph.tensors[tensor_idx])
            o2o.log(
                f"  Subgraph {subgraph_index_to_modify}: Removing tensor at index {tensor_idx}: {tensor_name}"
            )
            del subgraph.tensors[tensor_idx]
            removed_indices.append(tensor_idx)
        else:
            o2o.log(
                f"  Subgraph {subgraph_index_to_modify}: Warning: Tensor index {tensor_idx} out of bounds, skipping."
            )

    if not removed_indices:
        return []

    # Create a map for old index to new index after removal
    new_indices_map = {}
    current_new_idx = 0
    # Iterate over original tensor count of this subgraph
    original_tensor_count = len(subgraph.tensors) + len(removed_indices)
    for old_idx in range(original_tensor_count):
        if old_idx not in tensor_indices_to_remove:
            new_indices_map[old_idx] = current_new_idx
            current_new_idx += 1

    # Update operator inputs/outputs
    for op_idx, operator in enumerate(
            subgraph.operators):  # Object API: subgraph.operators
        if operator.inputs is not None:  # Object API: operator.inputs
            updated_inputs = []
            for j in range(len(operator.inputs)):  # Object API: len(operator.inputs)
                old_input_idx = operator.inputs[j]  # Object API: operator.inputs[j]
                if old_input_idx == -1:  # Optional empty input
                    updated_inputs.append(-1)
                elif old_input_idx in new_indices_map:
                    updated_inputs.append(new_indices_map[old_input_idx])
            operator.inputs = updated_inputs

        if operator.outputs is not None:  # Object API: operator.outputs
            updated_outputs = []
            for j in range(len(operator.outputs)):  # Object API: len(operator.outputs)
                old_output_idx = operator.outputs[j]  # Object API: operator.outputs[j]
                if old_output_idx in new_indices_map:
                    updated_outputs.append(new_indices_map[old_output_idx])
            operator.outputs = updated_outputs

        # Update intermediates if they exist
        if operator.intermediates is not None:  # Object API: operator.intermediates
            updated_intermediates = []
            for j in range(len(
                    operator.intermediates)):  # Object API: len(operator.intermediates)
                old_intermediate_idx = operator.intermediates[
                    j]  # Object API: operator.intermediates[j]
                if old_intermediate_idx in new_indices_map:
                    updated_intermediates.append(new_indices_map[old_intermediate_idx])
            operator.intermediates = updated_intermediates

    # Update subgraph inputs/outputs
    if subgraph.inputs is not None:  # Object API: subgraph.inputs
        updated_subgraph_inputs = []
        for j in range(len(subgraph.inputs)):  # Object API: len(subgraph.inputs)
            old_input_idx = subgraph.inputs[j]  # Object API: subgraph.inputs[j]
            if old_input_idx in new_indices_map:
                updated_subgraph_inputs.append(new_indices_map[old_input_idx])
        subgraph.inputs = updated_subgraph_inputs

    if subgraph.outputs is not None:  # Object API: subgraph.outputs
        updated_subgraph_outputs = []
        for j in range(len(subgraph.outputs)):  # Object API: len(subgraph.outputs)
            old_output_idx = subgraph.outputs[j]  # Object API: subgraph.outputs[j]
            if old_output_idx in new_indices_map:
                updated_subgraph_outputs.append(new_indices_map[old_output_idx])
        subgraph.outputs = updated_subgraph_outputs

    # Update SignatureDefs
    if model.signatureDefs:
        for sig_def in model.signatureDefs:
            if sig_def.subgraphIndex == subgraph_index_to_modify:
                # Update inputs
                if sig_def.inputs:
                    updated_sig_inputs = []
                    for tensor_map in sig_def.inputs:
                        if tensor_map.tensorIndex in new_indices_map:
                            tensor_map.tensorIndex = new_indices_map[
                                tensor_map.tensorIndex]
                            updated_sig_inputs.append(tensor_map)
                        elif tensor_map.tensorIndex in tensor_indices_to_remove:
                            o2o.log(
                                f"  SignatureDef '{sig_def.signatureKey}': Removing input tensor index {tensor_map.tensorIndex}"
                            )
                    sig_def.inputs = updated_sig_inputs

                # Update outputs
                if sig_def.outputs:
                    updated_sig_outputs = []
                    for tensor_map in sig_def.outputs:
                        if tensor_map.tensorIndex in new_indices_map:
                            tensor_map.tensorIndex = new_indices_map[
                                tensor_map.tensorIndex]
                            updated_sig_outputs.append(tensor_map)
                        elif tensor_map.tensorIndex in tensor_indices_to_remove:
                            o2o.log(
                                f"  SignatureDef '{sig_def.signatureKey}': Removing output tensor index {tensor_map.tensorIndex}"
                            )
                    sig_def.outputs = updated_sig_outputs

    return sorted(removed_indices)


def remove_buffers_and_update_model(model: ModelT,
                                    buffer_indices_to_remove: List[int]) -> List[int]:
    """
    Removes specified buffers from the model and updates all tensor references.
    This function uses the Object API for mutable model objects.

    Args:
        model: The mutable Circle model object (ModelT).
        buffer_indices_to_remove (list): A list of buffer indices to remove.
                                        Must be sorted in descending order.

    Returns:
        list: The list of buffer indices that were actually removed.
    """
    if not model.buffers:
        o2o.log("Model has no buffers to remove.")
        return []

    removed_indices = []

    # Sort in descending order to avoid index shifting issues during removal
    for buffer_idx in sorted(buffer_indices_to_remove, reverse=True):
        if 0 <= buffer_idx < len(model.buffers):
            o2o.log(f"  Removing buffer at index {buffer_idx}")
            del model.buffers[buffer_idx]
            removed_indices.append(buffer_idx)
        else:
            o2o.log(f"  Warning: Buffer index {buffer_idx} out of bounds, skipping.")

    if not removed_indices:
        return []

    # Create a map for old index to new index after removal
    new_indices_map = {}
    current_new_idx = 0
    # Iterate over original buffer count
    original_buffer_count = len(model.buffers) + len(removed_indices)
    for old_idx in range(original_buffer_count):
        if old_idx not in buffer_indices_to_remove:
            new_indices_map[old_idx] = current_new_idx
            current_new_idx += 1

    # Update tensor buffer references in all subgraphs
    for subgraph_idx, subgraph in enumerate(model.subgraphs):
        for tensor_idx, tensor in enumerate(subgraph.tensors):
            if tensor.buffer != -1:  # -1 indicates no buffer
                if tensor.buffer in new_indices_map:
                    old_buffer_idx = tensor.buffer
                    tensor.buffer = new_indices_map[old_buffer_idx]
                # If tensor.buffer was removed, set to -1 (no buffer)
                elif tensor.buffer in buffer_indices_to_remove:
                    tensor.buffer = -1

    return sorted(removed_indices)


def update_signature_defs_for_pruned_io(model: ModelT, subgraph_index: int,
                                        removed_inputs: List[int],
                                        removed_outputs: List[int]):
    """
    Updates SignatureDefs to remove references to pruned inputs/outputs.
    """
    if not model.signatureDefs:
        return

    for sig_def in model.signatureDefs:
        if sig_def.subgraphIndex == subgraph_index:
            # Update inputs
            if sig_def.inputs and removed_inputs:
                original_len = len(sig_def.inputs)
                sig_def.inputs = [
                    tm for tm in sig_def.inputs if tm.tensorIndex not in removed_inputs
                ]
                if len(sig_def.inputs) < original_len:
                    o2o.log(
                        f"  SignatureDef '{sig_def.signatureKey}': Pruned {original_len - len(sig_def.inputs)} input(s)"
                    )

            # Update outputs
            if sig_def.outputs and removed_outputs:
                original_len = len(sig_def.outputs)
                sig_def.outputs = [
                    tm for tm in sig_def.outputs if tm.tensorIndex not in removed_outputs
                ]
                if len(sig_def.outputs) < original_len:
                    o2o.log(
                        f"  SignatureDef '{sig_def.signatureKey}': Pruned {original_len - len(sig_def.outputs)} output(s)"
                    )


def prune_unused_io(model: ModelT) -> bool:
    """
    Removes tensors from Subgraph Inputs/Outputs if they are not connected to any operator.

    Args:
        model: The mutable Circle model object.

    Returns:
        bool: True if any changes were made.
    """
    changed = False
    for i, subgraph in enumerate(model.subgraphs):
        removed_inputs = []
        removed_outputs = []

        # Collect used inputs and outputs from operators
        op_inputs = set()
        op_outputs = set()
        for op_idx, op in enumerate(subgraph.operators):
            if op.inputs is not None:
                for inp in op.inputs:
                    if inp != -1:
                        op_inputs.add(inp)
            if op.outputs is not None:
                for out in op.outputs:
                    op_outputs.add(out)

        # Prune Subgraph Inputs
        # A Subgraph Input is unused if it is not consumed by any operator
        if subgraph.inputs is not None:
            original_len = len(subgraph.inputs)
            new_inputs = [idx for idx in subgraph.inputs if idx in op_inputs]
            if len(new_inputs) < original_len:
                removed_inputs = [idx for idx in subgraph.inputs if idx not in op_inputs]
                o2o.log(
                    f"Subgraph {i}: Pruning unused inputs (not consumed by any op): {removed_inputs}"
                )
                subgraph.inputs = new_inputs
                changed = True

        # Prune Subgraph Outputs
        # A Subgraph Output is unused if it is not produced by any operator
        if subgraph.outputs is not None:
            original_len = len(subgraph.outputs)
            new_outputs = [idx for idx in subgraph.outputs if idx in op_outputs]
            if len(new_outputs) < original_len:
                removed_outputs = [
                    idx for idx in subgraph.outputs if idx not in op_outputs
                ]
                o2o.log(
                    f"Subgraph {i}: Pruning unused outputs (not produced by any op): {removed_outputs}"
                )
                subgraph.outputs = new_outputs
                changed = True

        # Update SignatureDefs if any IO was pruned
        if removed_inputs or removed_outputs:
            update_signature_defs_for_pruned_io(model, i, removed_inputs, removed_outputs)

    return changed


def main():
    # Read the entire model from stdin
    data = sys.stdin.buffer.read()
    buf = bytearray(data)

    # Create a read‑only model (Native API) and a mutable copy (Object API)
    model_ro = circle.Model.GetRootAsModel(buf, 0)
    model = circle.ModelT.InitFromObj(model_ro)

    total_unused_tensors_count = 0
    model_changed = False

    # Prune unused inputs/outputs first
    if prune_unused_io(model):
        model_changed = True

    o2o.log(f"Processing {len(model.subgraphs)} subgraph(s)...")
    for i, subgraph in enumerate(model.subgraphs):
        # Use the mutable subgraph which might have been updated by prune_unused_io
        unused = find_unused_tensors_in_subgraph(subgraph)
        if not unused:
            o2o.log(f"Subgraph {i}: No unused tensors found.")
            continue

        total_unused_tensors_count += len(unused)
        o2o.log(
            f"Subgraph {i}: Found {len(unused)} unused tensor(s): {', '.join(map(str, sorted(unused)))}"
        )

        actually_removed = remove_tensors_and_update_model(model, i, unused)
        if actually_removed:
            o2o.log(f"Subgraph {i}: Removed {len(actually_removed)} tensor(s).")
            model_changed = True
        else:
            o2o.log(f"Subgraph {i}: No tensors were removed during the process.")

    if total_unused_tensors_count == 0:
        o2o.log("\nNo unused tensors found in any subgraph.")
        o2o.save_model_to_stdout(model)
        sys.exit(0)

    o2o.log(
        f"\nTotal unused tensors found across all subgraphs: {total_unused_tensors_count}"
    )

    # After removing tensors, now process unused buffers
    # Use the mutable model directly since find_unused_buffers now supports both APIs
    unused_buffers = find_unused_buffers(model)
    if unused_buffers:
        o2o.log(
            f"Found {len(unused_buffers)} unused buffer(s): {', '.join(map(str, sorted(unused_buffers)))}"
        )
        actually_removed_buffers = remove_buffers_and_update_model(model, unused_buffers)
        if actually_removed_buffers:
            o2o.log(f"Removed {len(actually_removed_buffers)} buffer(s).")
            model_changed = True
        else:
            o2o.log("No buffers were actually removed during the process.")
    else:
        o2o.log("No unused buffers found.")

    if model_changed:
        o2o.log("\nSaving modified model to stdout...")
    else:
        o2o.log(
            "\nNo tensors or buffers were actually removed. Saving original model to stdout."
        )
    o2o.save_model_to_stdout(model)


if __name__ == "__main__":
    main()
