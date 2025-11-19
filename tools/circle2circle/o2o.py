#!/usr/bin/env python3

import sys
import circle
import flatbuffers

# ============================================================================
# BASIC UTILITIES
# ============================================================================


def log(message):
    """Log message to stderr"""
    print(message, file=sys.stderr)


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


# ============================================================================
# CORE I/O FUNCTIONS
# ============================================================================


def load_circle_model(input_file=None):
    """Load and parse a circle model file"""
    if input_file is None:
        # Read from stdin
        data = sys.stdin.buffer.read()
    else:
        # Read from file
        with open(input_file, 'rb') as f:
            data = f.read()

    buf = bytearray(data)
    model = circle.Model.GetRootAsModel(buf, 0)
    model = circle.ModelT.InitFromObj(model)
    return model


def load_model_from_stdin():
    """Load a Circle model from binary data read from stdin."""
    return load_circle_model()  # input_file=None defaults to stdin


def save_circle_model(model, output_file=None):
    """Save a circle model to file using flatbuffers"""
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), b'CIR0')

    if output_file is None:
        # Write to stdout
        sys.stdout.buffer.write(builder.Output())
    else:
        # Write to file
        with open(output_file, 'wb') as f:
            f.write(builder.Output())


def save_model_to_stdout(model):
    """Serialize a Circle model and write it to stdout as binary data."""
    save_circle_model(model)  # output_file=None defaults to stdout


# ============================================================================
# CLI HANDLING
# ============================================================================


def handle_cli_args(usage_message):
    """Handle common command line argument parsing and validation"""
    if len(sys.argv) != 3:
        log(usage_message)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    return input_file, output_file


# ============================================================================
# TENSOR UTILITIES
# ============================================================================


def get_tensor_name(tensor):
    """Get tensor name as string, handling bytes conversion"""
    if tensor.name:
        return tensor.name.decode('utf-8') if isinstance(tensor.name,
                                                         bytes) else tensor.name
    return None


def get_tensor_by_index(subgraph, index):
    """Safely get tensor by its index."""
    if 0 <= index < len(subgraph.tensors):
        return subgraph.tensors[index]
    return None


def get_tensor_index_by_name(subgraph, name):
    """Find tensor index by name, handling byte strings."""
    name_bytes = name.encode('utf-8')  # Convert str to bytes for comparison
    for i, tensor in enumerate(subgraph.tensors):
        if tensor.name and tensor.name == name_bytes:
            return i
    return -1  # Not found


def is_tensor_constant(tensor, model_buffers):
    """Check if a tensor is constant by verifying its buffer."""
    if tensor and tensor.buffer != 0 and 0 <= tensor.buffer - 1 < len(model_buffers):
        # A non-zero buffer index that points to a valid buffer typically means it's constant.
        # The 0th buffer is always an empty buffer.
        return True
    return False


# ============================================================================
# TENSOR PROCESSING FUNCTIONS
# ============================================================================


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


# ============================================================================
# OPERATOR UTILITIES
# ============================================================================


def parse_operator_indices(indices_str):
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
                log(f"Error parsing range '{part}': {e}")
                sys.exit(1)
        else:
            # Single index
            try:
                idx = int(part)
                if idx < 0:
                    raise ValueError("Index must be non-negative")
                indices.add(idx)
            except ValueError as e:
                log(f"Error parsing index '{part}': {e}")
                sys.exit(1)

    return sorted(list(indices))


def get_or_create_operator_code(model, builtin_op_type):
    """Get the index of an operator code, or create it if it doesn't exist."""
    for i, op_code in enumerate(model.operatorCodes):
        if op_code.builtinCode == builtin_op_type:
            return i

    # If not found, create a new one
    new_op_code = circle.OperatorCodeT()
    new_op_code.builtinCode = builtin_op_type
    new_op_code.deprecatedBuiltinCode = builtin_op_type
    new_op_code.version = 1  # Default version
    model.operatorCodes.append(new_op_code)
    return len(model.operatorCodes) - 1
