#!/usr/bin/env python3

import sys
# import argparse  # Removed: script now uses stdin/stdout instead of file arguments
import flatbuffers
import circle
import o2o  # For saving the model


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

    if model_changed:
        o2o.log("\nSaving modified model to stdout...")
    else:
        o2o.log(
            "\nNo tensors were actually removed from any subgraph. Saving original model to stdout."
        )
    o2o.save_model_to_stdout(model)


if __name__ == "__main__":
    main()
