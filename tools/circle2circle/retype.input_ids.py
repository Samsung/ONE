#!/usr/bin/env python3

import o2o
import circle
import sys


def retype_input_ids():
    """Main function to change input_ids tensor type from int64 to int32"""
    # Load the model using utility function
    model = o2o.load_model_from_stdin()

    def process_subgraph(subgraph):
        """Process a single subgraph to find and retype input_ids tensors"""
        o2o.log(f"Processing subgraph with {len(subgraph.tensors)} tensors")

        retyped_count = 0
        for tensor in subgraph.tensors:
            tensor_name = o2o.get_tensor_name(tensor)

            # Check if this is the input_ids tensor
            if tensor_name == "input_ids":
                # Check if current type is int64
                if tensor.type == circle.TensorType.INT64:
                    old_type = "int64"
                    new_type = "int32"

                    # Change type to int32
                    tensor.type = circle.TensorType.INT32

                    o2o.log(f"Retyped tensor: {tensor_name} {old_type} → {new_type}")
                    retyped_count += 1
                else:
                    o2o.log(
                        f"Found input_ids tensor but type is not int64 (current type: {tensor.type})"
                    )

        if retyped_count > 0:
            o2o.log(f"Retyped {retyped_count} input_ids tensors in this subgraph")
        else:
            o2o.log("No input_ids tensors were retyped in this subgraph")

        return retyped_count > 0, retyped_count

    # Process all subgraphs using utility function
    overall_modified, total_changes = o2o.process_subgraphs(model, process_subgraph)

    if not overall_modified:
        o2o.log("No input_ids tensors were modified.")

    # Save the model using utility function
    o2o.save_model_to_stdout(model)


if __name__ == "__main__":
    # Directly invoke processing; I/O handled via stdin/stdout
    retype_input_ids()
