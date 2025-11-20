#!/usr/bin/env python3

import o2o
import re
import sys

# Import specific Circle types for better type annotations
from circle import (TensorT, OperatorT, SubGraphT, ModelT, BufferT, OperatorCodeT,
                    BuiltinOperator, TensorType)


def rename_input_tensors(prefix: str):
    """Main function to rename tensors by removing the specified prefix"""
    # Load the model using utility function
    model = o2o.load_model_from_stdin()

    # Pattern to match tensor names that start with the specified prefix followed by anything
    pattern = re.escape(prefix) + r'(.*)'

    def process_subgraph(subgraph):
        """Process a single subgraph"""
        o2o.log(f"Processing subgraph with {len(subgraph.tensors)} tensors")

        renamed_count = 0
        for tensor in subgraph.tensors:
            was_renamed, old_name, new_name = o2o.rename_tensor_if_matches(
                tensor, pattern, lambda match: match.group(1))

            if was_renamed:
                o2o.log(f"Renaming tensor: {old_name} → {new_name}")
                renamed_count += 1

        if renamed_count > 0:
            o2o.log(f"Renamed {renamed_count} tensors in this subgraph")
        else:
            o2o.log("No tensors were renamed in this subgraph")

        return renamed_count > 0, renamed_count

    # Process all subgraphs using utility function
    overall_modified, total_changes = o2o.process_subgraphs(model, process_subgraph)

    if not overall_modified:
        o2o.log("No tensors were modified.")

    # Save the model using utility function
    o2o.save_model_to_stdout(model)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        o2o.log("Usage: python rename_inputs.py <prefix>")
        sys.exit(1)

    prefix = sys.argv[1]

    # Directly invoke processing; I/O handled via stdin/stdout
    rename_input_tensors(prefix)
