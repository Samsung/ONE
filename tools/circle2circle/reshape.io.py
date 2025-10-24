#!/usr/bin/env python3

import sys
import argparse
import circle
import o2o


def parse_shape(shape_str):
    """Parse a shape string like '[1,16,30,4]' into a list of integers."""
    try:
        # Strip surrounding brackets and whitespace, then split by commas
        shape_str = shape_str.strip().strip('[]')
        parts = shape_str.split(',')
        shape = [int(p.strip()) for p in parts if p.strip()]
        return shape
    except Exception as e:
        raise ValueError(
            f"Invalid shape format '{shape_str}'. Expected format like [1,16,30,4]"
        ) from e


def is_target_shape(shape, target_shape):
    """Check if a tensor shape matches the target shape."""
    if len(shape) != len(target_shape):
        return False
    return list(shape) == target_shape


def reshape_input_tensors(io_type, target_shape, new_shape):
    """Reshape input or output tensors from target_shape to new_shape."""
    model = o2o.load_model_from_stdin()

    for subgraph in model.subgraphs:
        # Choose the appropriate tensor list based on io_type
        io_list = subgraph.inputs if io_type == 'input' else subgraph.outputs

        for tensor_idx in io_list:
            tensor = subgraph.tensors[tensor_idx]
            if is_target_shape(tensor.shape, target_shape):
                tensor.shape = new_shape

    o2o.save_model_to_stdout(model)


def main():
    parser = argparse.ArgumentParser(
        description='Reshape input or output tensors by specifying target and new shapes.'
    )
    parser.add_argument('io_type',
                        choices=['input', 'output'],
                        help='Whether to process input tensors or output tensors.')
    parser.add_argument(
        '--by_shape',
        nargs=2,
        metavar=('TARGET_SHAPE', 'NEW_SHAPE'),
        required=True,
        help=
        'Reshape tensors from TARGET_SHAPE to NEW_SHAPE. Example: --by_shape [1,16,30,4] [1,16,32,4]'
    )
    # No file arguments needed; model is read from stdin and written to stdout

    args = parser.parse_args()

    # Parse the shape arguments
    try:
        target_shape = parse_shape(args.by_shape[0])
        new_shape = parse_shape(args.by_shape[1])
    except ValueError as e:
        o2o.log(f"Error parsing shapes: {e}")
        sys.exit(1)

    # Execute the reshaping with safe error handling
    reshape_input_tensors(args.io_type, target_shape, new_shape)


if __name__ == "__main__":
    main()
