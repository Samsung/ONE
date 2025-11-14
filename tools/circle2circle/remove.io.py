#!/usr/bin/env python3

import sys
import argparse
import o2o


def parse_names(names_str):
    """Parse comma‑separated tensor names into a list of names."""
    return [name.strip() for name in names_str.split(',') if name.strip()]


def remove_io_tensors(io_type, names_to_keep):
    """Remove input or output tensors, keeping only specified tensor names"""
    # Load the model using utility function
    model = o2o.load_model_from_stdin()

    def process_subgraph(subgraph):
        """Process a single subgraph"""
        if io_type == 'input':
            io_list = subgraph.inputs
            io_name = 'input'
        elif io_type == 'output':
            io_list = subgraph.outputs
            io_name = 'output'
        else:
            raise ValueError(f"Invalid io_type: {io_type}. Must be 'input' or 'output'")

        o2o.log(f"Processing subgraph with {len(io_list)} {io_name}s")
        o2o.log(f"Original {io_name} indices: {io_list}")

        # Build a mapping from tensor name to its index for the selected I/O list
        name_to_index = {}
        for io_idx in io_list:
            tensor = subgraph.tensors[io_idx]
            tensor_name = o2o.get_tensor_name(tensor)
            if tensor_name:
                name_to_index[tensor_name] = io_idx

        # Filter tensors to keep by name
        new_io_list = []
        for name in names_to_keep:
            if name in name_to_index:
                new_io_list.append(name_to_index[name])
            else:
                o2o.log(f"Warning: {io_name} tensor name '{name}' not found")

        # Update the subgraph
        if io_type == 'input':
            subgraph.inputs = new_io_list
        else:
            subgraph.outputs = new_io_list

        o2o.log(f"New {io_name} indices: {[i+1 for i in range(len(new_io_list))]}")

        removed_count = len(io_list) - len(new_io_list)
        return removed_count > 0, removed_count

    # Process all subgraphs using utility function
    overall_modified, total_changes = o2o.process_subgraphs(model, process_subgraph)

    if not overall_modified:
        o2o.log("No tensors were removed.")

    # Save the model using utility function
    o2o.save_model_to_stdout(model)

def remove_io_tensors_by_id(io_type, ids_to_keep):
    """Remove input or output tensors, keeping only specified tensor indices (IDs)"""
    model = o2o.load_model_from_stdin()

    def process_subgraph(subgraph):
        if io_type == 'input':
            io_list = subgraph.inputs
            io_name = 'input'
        elif io_type == 'output':
            io_list = subgraph.outputs
            io_name = 'output'
        else:
            raise ValueError(f"Invalid io_type: {io_type}. Must be 'input' or 'output'")

        o2o.log(f"Processing subgraph with {len(io_list)} {io_name}s")
        o2o.log(f"Original {io_name} indices: {io_list}")

        # Keep only those indices whose position in the original list matches ids_to_keep
        new_io_list = []
        for idx, tensor_idx in enumerate(io_list):
            if idx in ids_to_keep:
                new_io_list.append(tensor_idx)
            else:
                o2o.log(f"Removing {io_name} tensor at position {idx}")

        # Update the subgraph
        if io_type == 'input':
            subgraph.inputs = new_io_list
        else:
            subgraph.outputs = new_io_list

        o2o.log(f"New {io_name} indices: {new_io_list}")

        removed_count = len(io_list) - len(new_io_list)
        return removed_count > 0, removed_count

    overall_modified, total_changes = o2o.process_subgraphs(model, process_subgraph)

    if not overall_modified:
        o2o.log("No tensors were removed.")

    o2o.save_model_to_stdout(model)


def main():
    parser = argparse.ArgumentParser(
        description=
        'Remove input or output tensors from Circle model, keeping only specified tensor names or IDs'
    )
    parser.add_argument('io_type',
                        choices=['input', 'output'],
                        help='Whether to process inputs or outputs')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--keep_by_name',
        help='Comma‑separated tensor names to keep (e.g., "tensorA,tensorB")')
    group.add_argument(
        '--keep_by_id',
        help='Comma‑separated tensor IDs or ranges to keep (e.g., "0,2-4")')
    # No file arguments needed; model is read from stdin and written to stdout

    args = parser.parse_args()

    if args.keep_by_name:
        # Parse the tensor names
        try:
            names_to_keep = parse_names(args.keep_by_name)
            o2o.log(f"Tensor names to keep: {names_to_keep}")
        except ValueError as e:
            o2o.log(f"Error parsing tensor names: {e}")
            sys.exit(1)
        # Execute name‑based removal
        remove_io_tensors(args.io_type, names_to_keep)
    elif args.keep_by_id:
        # Parse the tensor IDs
        try:
            ids_to_keep = o2o.parse_operator_indices(args.keep_by_id)
            o2o.log(f"Tensor IDs to keep: {ids_to_keep}")
        except Exception as e:
            o2o.log(f"Error parsing tensor IDs: {e}")
            sys.exit(1)
        # Execute ID‑based removal
        remove_io_tensors_by_id(args.io_type, ids_to_keep)


if __name__ == "__main__":
    main()
