#!/usr/bin/env python3

import sys
import argparse
from typing import List, Dict, Tuple
import o2o

# Import specific Circle types for better type annotations
from circle import (TensorT, OperatorT, SubGraphT, ModelT, BufferT, OperatorCodeT,
                    BuiltinOperator, TensorType)


def parse_operator_indices(indices_str: str) -> List[int]:
    """Parse operator index string into a list of indices.

    Supports formats like:
    - "0-181" (range)
    - "0,5,10-15" (mixed)
    - "0" (single index)

    Args:
        indices_str (str): String containing operator indices

    Returns:
        list: Sorted list of unique operator indices
    """
    if not indices_str:
        return []

    indices = set()

    # Split by comma first
    parts = indices_str.split(',')

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Check if it's a range
        if '-' in part:
            try:
                start, end = part.split('-', 1)
                start_idx = int(start.strip())
                end_idx = int(end.strip())

                if start_idx < 0 or end_idx < 0:
                    raise ValueError("Indices must be non-negative")

                if start_idx > end_idx:
                    raise ValueError(f"Invalid range: {start_idx} > {end_idx}")

                indices.update(range(start_idx, end_idx + 1))
            except ValueError as e:
                o2o.log(f"Error parsing range '{part}': {e}")
                sys.exit(1)
        else:
            # Single index
            try:
                idx = int(part)
                if idx < 0:
                    raise ValueError("Index must be non-negative")
                indices.add(idx)
            except ValueError as e:
                o2o.log(f"Error parsing index '{part}': {e}")
                sys.exit(1)

    return sorted(list(indices))


def analyze_tensor_connections(subgraph: SubGraphT) -> Dict[str, any]:
    """Analyze all tensor connections in the subgraph.

    Args:
        subgraph: Circle subgraph object

    Returns:
        dict: Analysis results including tensor-to-operator mappings and subgraph I/O info
    """
    # Build mappings
    tensor_to_def = {}  # tensor_idx -> operator_idx
    tensor_to_use = {}  # tensor_idx -> [operator_idx, ...]
    op_inputs = {}  # operator_idx -> [tensor_idx, ...]
    op_outputs = {}  # operator_idx -> [tensor_idx, ...]

    # Analyze all operators
    for op_idx, operator in enumerate(subgraph.operators):
        inputs = []
        if operator.inputs is not None and len(operator.inputs) > 0:
            inputs = list(operator.inputs)
        outputs = []
        if operator.outputs is not None and len(operator.outputs) > 0:
            outputs = list(operator.outputs)

        op_inputs[op_idx] = inputs
        op_outputs[op_idx] = outputs

        # Record tensor -> producer mapping
        for output_idx in outputs:
            if output_idx != -1:
                tensor_to_def[output_idx] = op_idx

        # Record tensor -> consumers mapping
        for input_idx in inputs:
            if input_idx != -1:
                if input_idx not in tensor_to_use:
                    tensor_to_use[input_idx] = []
                tensor_to_use[input_idx].append(op_idx)

    # Analyze subgraph I/O
    subgraph_inputs = list(subgraph.inputs) if subgraph.inputs is not None else []
    subgraph_outputs = list(subgraph.outputs) if subgraph.outputs is not None else []

    return {
        'tensor_to_producer': tensor_to_def,
        'tensor_to_consumers': tensor_to_use,
        'operator_to_inputs': op_inputs,
        'operator_to_outputs': op_outputs,
        'subgraph_inputs': subgraph_inputs,
        'subgraph_outputs': subgraph_outputs
    }


def select_operators_and_update_model(
        model: ModelT, subgraph_index: int,
        operator_indices_to_keep: List[int]) -> Tuple[int, int]:
    """Keep only specified operators in the model and remove all others.

    Args:
        model: Circle model object (Object API)
        subgraph_index (int): Index of subgraph to modify (assumed to be 0)
        operator_indices_to_keep (list): List of operator indices to keep

    Returns:
        tuple: (removed_operators_count, removed_operator_codes_count)
    """
    if not model.subgraphs or subgraph_index >= len(model.subgraphs):
        o2o.log(f"Error: Invalid subgraph index {subgraph_index}")
        return 0, 0

    subgraph = model.subgraphs[subgraph_index]

    # Validate operator indices
    max_operator_index = len(subgraph.operators) - 1
    invalid_indices = [
        idx for idx in operator_indices_to_keep if idx > max_operator_index
    ]
    if invalid_indices:
        o2o.log(
            f"Error: Operator indices {invalid_indices} exceed maximum index {max_operator_index}"
        )
        sys.exit(1)

    o2o.log(
        f"Subgraph {subgraph_index}: Keeping {len(operator_indices_to_keep)} operator(s): {operator_indices_to_keep}"
    )

    # Step 1: Determine which operators to remove
    total_operators = len(subgraph.operators)
    operator_indices_to_remove = []
    for i in range(total_operators):
        if i not in operator_indices_to_keep:
            operator_indices_to_remove.append(i)

    o2o.log(
        f"Will remove {len(operator_indices_to_remove)} operator(s): {operator_indices_to_remove}"
    )

    # Step 2: Analyze tensor connections BEFORE removing operators
    connections = analyze_tensor_connections(subgraph)

    # Step 3: Remove operators in descending order to avoid index shifting
    removed_operators = []
    for op_idx in sorted(operator_indices_to_remove, reverse=True):
        del subgraph.operators[op_idx]
        removed_operators.append(op_idx)

    # Step 4: Update subgraph I/O
    # Remove subgraph inputs that were used only by removed operators
    inputs_to_remove = set()
    for input_idx in connections['subgraph_inputs']:
        if input_idx in connections['tensor_to_consumers']:
            # Check if all consumers of this input were removed
            all_consumers_removed = True
            for consumer_idx in connections['tensor_to_consumers'][input_idx]:
                if consumer_idx not in operator_indices_to_remove:
                    all_consumers_removed = False
                    break
            if all_consumers_removed:
                inputs_to_remove.add(input_idx)

    # Remove subgraph outputs that were produced only by removed operators
    outputs_to_remove = set()
    for output_idx in connections['subgraph_outputs']:
        if output_idx in connections['tensor_to_producer']:
            if connections['tensor_to_producer'][
                    output_idx] in operator_indices_to_remove:
                outputs_to_remove.add(output_idx)

    # Update subgraph inputs
    if inputs_to_remove:
        new_inputs = [
            idx for idx in connections['subgraph_inputs'] if idx not in inputs_to_remove
        ]
        subgraph.inputs = new_inputs
        o2o.log(
            f"Removed {len(inputs_to_remove)} subgraph inputs: {sorted(inputs_to_remove)}"
        )

    # Update subgraph outputs
    if outputs_to_remove:
        new_outputs = [
            idx for idx in connections['subgraph_outputs'] if idx not in outputs_to_remove
        ]
        subgraph.outputs = new_outputs
        o2o.log(
            f"Removed {len(outputs_to_remove)} subgraph outputs: {sorted(outputs_to_remove)}"
        )

    # Step 5: Update operator inputs that reference outputs of removed operators
    for op_idx, operator in enumerate(subgraph.operators):
        if operator.inputs is not None and len(operator.inputs) > 0:
            updated_inputs = []
            for input_idx in operator.inputs:
                if input_idx != -1 and input_idx in connections['tensor_to_producer']:
                    producer_idx = connections['tensor_to_producer'][input_idx]
                    if producer_idx in operator_indices_to_remove:
                        # This input comes from a removed operator, set to -1
                        updated_inputs.append(-1)
                        o2o.log(
                            f"  Operator {op_idx}: Breaking input connection from removed operator {producer_idx}"
                        )
                    else:
                        updated_inputs.append(input_idx)
                else:
                    updated_inputs.append(input_idx)
            operator.inputs = updated_inputs

    # Step 6: Clean up unused OperatorCode entries
    # Get OperatorCode usage by remaining operators
    used_operator_codes = set()
    for operator in subgraph.operators:
        if operator.opcodeIndex is not None:
            used_operator_codes.add(operator.opcodeIndex)

    # Find unused OperatorCode indices
    unused_operator_codes = []
    for i, operator_code in enumerate(model.operatorCodes):
        if i not in used_operator_codes:
            unused_operator_codes.append(i)

    # Remove unused OperatorCode entries in descending order
    removed_operator_codes = []
    for code_idx in sorted(unused_operator_codes, reverse=True):
        operator_code = model.operatorCodes[code_idx]
        if operator_code.builtinCode is not None:
            op_name = f"builtin_code={operator_code.builtinCode}"
        else:
            op_name = f"custom_code={operator_code.customCode}"
        o2o.log(f"  Removing unused OperatorCode at index {code_idx}: {op_name}")
        del model.operatorCodes[code_idx]
        removed_operator_codes.append(code_idx)

    # Step 7: Update operator code indices in remaining operators
    # Create mapping from old to new indices
    old_to_new_code_indices = {}
    new_idx = 0
    for old_idx in range(len(model.operatorCodes) + len(removed_operator_codes)):
        if old_idx not in unused_operator_codes:
            old_to_new_code_indices[old_idx] = new_idx
            new_idx += 1

    # Update operator code indices
    for subgraph in model.subgraphs:
        for operator in subgraph.operators:
            if operator.opcodeIndex is not None:
                old_code_idx = operator.opcodeIndex
                if old_code_idx in old_to_new_code_indices:
                    operator.opcodeIndex = old_to_new_code_indices[old_code_idx]

    # Count tensor usage and definition
    tensor_use_count = {}
    tensor_def_count = {}

    # Count usage (inputs)
    for operator in subgraph.operators:
        if operator.inputs is not None:
            for input_idx in operator.inputs:
                if input_idx != -1:
                    tensor_use_count[input_idx] = tensor_use_count.get(input_idx, 0) + 1

    # Count definition (outputs)
    for operator in subgraph.operators:
        if operator.outputs is not None:
            for output_idx in operator.outputs:
                if output_idx != -1:
                    tensor_def_count[output_idx] = tensor_def_count.get(output_idx, 0) + 1

    # Add tensors with use_count == 0 to subgraph outputs
    added_outputs = []
    current_outputs = set(subgraph.outputs) if subgraph.outputs is not None else set()

    # First check tensors that are in tensor_use_count
    for tensor_idx, use_count in tensor_use_count.items():
        if use_count == 0 and tensor_idx not in current_outputs:
            subgraph.outputs = list(
                subgraph.outputs) if subgraph.outputs is not None else []
            subgraph.outputs.append(tensor_idx)
            added_outputs.append(tensor_idx)

    # Then check all output tensors from operators (some might not be in tensor_use_count)
    for operator in subgraph.operators:
        if operator.outputs is not None:
            for output_idx in operator.outputs:
                if output_idx != -1:
                    use_count = tensor_use_count.get(
                        output_idx, 0)  # Default to 0 if not in use_count
                    if use_count == 0 and output_idx not in current_outputs and output_idx not in added_outputs:
                        subgraph.outputs = list(
                            subgraph.outputs) if subgraph.outputs is not None else []
                        subgraph.outputs.append(output_idx)
                        added_outputs.append(output_idx)

    if added_outputs:
        o2o.log(f"Added tensors to subgraph outputs: {sorted(added_outputs)}")

    # Add tensors with def_count == 0 to subgraph inputs
    added_inputs = []
    current_inputs = set(subgraph.inputs) if subgraph.inputs is not None else set()
    for tensor_idx, def_count in tensor_def_count.items():
        if def_count == 0 and tensor_idx not in current_inputs:
            subgraph.inputs = list(subgraph.inputs) if subgraph.inputs is not None else []
            subgraph.inputs.append(tensor_idx)
            added_inputs.append(tensor_idx)

    if added_inputs:
        o2o.log(f"Added tensors to subgraph inputs: {sorted(added_inputs)}")

    return len(removed_operators), len(removed_operator_codes)


def main():
    parser = argparse.ArgumentParser(
        description=
        'Select operators from Circle model by index range, then clean up unused tensors')
    parser.add_argument('--by_id',
                        required=True,
                        help='Operator indices to keep (e.g., "0-181", "0,5,10-15")')

    args = parser.parse_args()

    # Parse the operator indices
    try:
        operator_indices_to_keep = parse_operator_indices(args.by_id)
        o2o.log(f"Operator indices to keep: {operator_indices_to_keep}")
    except ValueError as e:
        o2o.log(f"Error parsing operator indices: {e}")
        sys.exit(1)

    if not operator_indices_to_keep:
        o2o.log("No valid operator indices specified")
        sys.exit(1)

    # Load the model
    model = o2o.load_model_from_stdin()

    # Assume only one subgraph (index 0)
    subgraph_index = 0

    if not model.subgraphs or subgraph_index >= len(model.subgraphs):
        o2o.log(f"Error: Model has no subgraph at index {subgraph_index}")
        sys.exit(1)

    o2o.log(f"Model has {len(model.subgraphs[subgraph_index].operators)} operators")

    # Select operators (keep only specified ones)
    removed_ops_count, removed_codes_count = select_operators_and_update_model(
        model, subgraph_index, operator_indices_to_keep)

    o2o.log(
        f"Removed {removed_ops_count} operators and {removed_codes_count} unused OperatorCode entries"
    )

    # Save the model directly to stdout
    o2o.save_model_to_stdout(model)


if __name__ == "__main__":
    main()
