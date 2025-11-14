#!/usr/bin/env python3

import sys
import circle
import o2o


def get_tensor_name(tensor):
    """Get tensor name as string, handling bytes conversion"""
    if tensor.name:
        return tensor.name.decode('utf-8') if isinstance(tensor.name,
                                                         bytes) else tensor.name
    return None


def find_unused_tensors_in_subgraph(subgraph):
    """
    Finds and returns the indices of unused tensors in a given subgraph.
    This function uses the Native API for read-only subgraph objects.

    Args:
        subgraph: The Circle read-only subgraph object.

    Returns:
        list: A list of integer indices representing unused tensors.
    """
    num_tensors = subgraph.TensorsLength()
    if num_tensors == 0:
        return []

    used_tensor_indices = set()
    output_tensor_indices = set()

    # Collect output tensor indices
    for i in range(subgraph.OutputsLength()):
        output_tensor_indices.add(subgraph.Outputs(i))

    # Collect input tensor indices from all operators
    for i in range(subgraph.OperatorsLength()):
        operator = subgraph.Operators(i)
        if operator and operator.InputsLength():
            for j in range(operator.InputsLength()):
                input_tensor_index = operator.Inputs(j)
                # In Circle schema, -1 indicates an optional input that is not used.
                if input_tensor_index != -1:
                    used_tensor_indices.add(input_tensor_index)

    # A tensor is unused if it's not used by any operator AND not an output of the subgraph
    unused_indices = []
    for i in range(num_tensors):
        if i not in used_tensor_indices and i not in output_tensor_indices:
            unused_indices.append(i)

    return unused_indices


def find_unused_buffers(model):
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


def remove_tensors_and_update_model(model, subgraph_index_to_modify,
                                    tensor_indices_to_remove):
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

    return sorted(removed_indices)


def remove_buffers_and_update_model(model, buffer_indices_to_remove):
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


def main():
    # Read the entire model from stdin
    data = sys.stdin.buffer.read()
    buf = bytearray(data)

    # Create a read‑only model (Native API) and a mutable copy (Object API)
    model_ro = circle.Model.GetRootAsModel(buf, 0)
    model = circle.ModelT.InitFromObj(model_ro)

    total_unused_tensors_count = 0
    model_changed = False

    o2o.log(f"Processing {model_ro.SubgraphsLength()} subgraph(s)...")
    for i in range(model_ro.SubgraphsLength()):
        subgraph_ro = model_ro.Subgraphs(i)
        if not subgraph_ro:
            o2o.log(f"Warning: Could not read subgraph {i}. Skipping.")
            continue

        unused = find_unused_tensors_in_subgraph(subgraph_ro)
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
