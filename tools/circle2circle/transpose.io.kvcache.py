#!/usr/bin/env python3

import o2o
import re


def transpose_2d_3d(shape):
    """Transpose the second and third dimensions of a 4D shape"""
    if len(shape) != 4:
        raise ValueError("Shape must be 4D to transpose second and third dimensions")
    # Transpose shape: [d0, d1, d2, d3] -> [d0, d2, d1, d3]
    return [shape[0], shape[2], shape[1], shape[3]]


def transpose_tensor_dimensions():
    """Main function to find tensors and transpose their dimensions"""
    # Load the model using utility function
    model = o2o.load_model_from_stdin()

    # Pattern to match tensor names like "past_key_values_key_cache_0", "past_key_values_key_cache_1", etc.
    pattern = r'.*_cache_\d+'

    def process_subgraph(subgraph):
        """Process a single subgraph"""
        o2o.log(f"Processing subgraph with {len(subgraph.inputs)} input tensors")

        modified_count = 0
        for input_tensor_index in subgraph.inputs:
            # Get the actual tensor object
            tensor = subgraph.tensors[input_tensor_index]

            tensor_name = o2o.get_tensor_name(tensor)
            if tensor_name and re.match(pattern, tensor_name):
                o2o.log(f"Found input tensor: {tensor_name} with shape {tensor.shape}")

                if len(tensor.shape) == 4:
                    o2o.log(
                        f"Input tensor {tensor_name} is 4D. Transposing second and third dimensions."
                    )

                    # Transpose the second and third dimensions
                    original_shape = tensor.shape.copy()
                    new_shape = transpose_2d_3d(tensor.shape)
                    tensor.shape = new_shape

                    o2o.log(f"Shape changed from {original_shape} to {new_shape}")
                    modified_count += 1
                else:
                    o2o.log(f"Input tensor {tensor_name} is not 4D. Skipping.")

        return modified_count > 0, modified_count

    # Process all subgraphs using utility function
    overall_modified, total_changes = o2o.process_subgraphs(model, process_subgraph)

    if not overall_modified:
        o2o.log("No tensors were modified.")

    # Save the model using utility function
    o2o.save_model_to_stdout(model)


if __name__ == "__main__":
    # Directly invoke processing; I/O handled via stdin/stdout
    transpose_tensor_dimensions()
