#!/usr/bin/env python3

import sys
import circle
import flatbuffers


def log(message):
    """Log message to stderr"""
    print(message, file=sys.stderr)


def load_model_from_stdin():
    """Load a Circle model from binary data read from stdin."""
    data = sys.stdin.buffer.read()
    buf = bytearray(data)
    model = circle.Model.GetRootAsModel(buf, 0)
    model = circle.ModelT.InitFromObj(model)
    return model


def save_model_to_stdout(model):
    """Serialize a Circle model and write it to stdout as binary data."""
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), b'CIR0')
    sys.stdout.buffer.write(builder.Output())


def load_circle_model(input_file):
    """Load and parse a circle model file"""
    with open(input_file, 'rb') as f:
        buf = bytearray(f.read())

    model = circle.Model.GetRootAsModel(buf, 0)
    model = circle.ModelT.InitFromObj(model)
    return model


def save_circle_model(model, output_file):
    """Save a circle model to file using flatbuffers"""
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), b'CIR0')

    with open(output_file, 'wb') as f:
        f.write(builder.Output())


def handle_cli_args(usage_message):
    """Handle common command line argument parsing and validation"""
    if len(sys.argv) != 3:
        log(usage_message)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    return input_file, output_file


def get_tensor_name(tensor):
    """Get tensor name as string, handling bytes conversion"""
    if tensor.name:
        return tensor.name.decode('utf-8') if isinstance(tensor.name,
                                                         bytes) else tensor.name
    return None


def process_subgraphs(model, processor_func):
    """Generic subgraph processor with modification tracking

    Args:
        model: Circle model object
        processor_func: Function that processes a subgraph and returns (modified, changes_count)

    Returns:
        tuple: (overall_modified, total_changes)
    """
    overall_modified = False
    total_changes = 0

    for subgraph in model.subgraphs:
        modified, changes_count = processor_func(subgraph)
        overall_modified = overall_modified or modified
        total_changes += changes_count

    return overall_modified, total_changes


def rename_tensor_if_matches(tensor, pattern, replacement_func):
    """Rename tensor if it matches the given pattern

    Args:
        tensor: Tensor object to process
        pattern: Regex pattern to match
        replacement_func: Function that takes regex match and returns new name

    Returns:
        tuple: (was_renamed, old_name, new_name)
    """
    tensor_name = get_tensor_name(tensor)
    if not tensor_name:
        return False, None, None

    import re
    match = re.match(pattern, tensor_name)
    if match:
        old_name = tensor_name
        new_name = replacement_func(match)
        tensor.name = new_name
        return True, old_name, new_name

    return False, None, None


def safe_execute(main_func,
                 input_file,
                 output_file,
                 *args,
                 error_message="Error processing file"):
    """Safely execute the main function with error handling"""
    try:
        main_func(input_file, output_file, *args)
        log(f"Successfully processed {input_file} and saved to {output_file}")
    except Exception as e:
        log(f"{error_message}: {e}")
        sys.exit(1)
