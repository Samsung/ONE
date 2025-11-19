#!/usr/bin/env python3

import sys
import argparse
import o2o
import circle


def get_operator_code_key(op_code):
    """Generate a unique key for an OperatorCode to identify duplicates.

    Args:
        op_code: Circle OperatorCode object

    Returns:
        tuple: Unique key for the operator code
    """
    if op_code.builtinCode is not None:
        # Builtin operator
        return ('builtin', op_code.builtinCode)
    elif op_code.customCode is not None:
        # Custom operator
        custom_code = op_code.customCode
        if isinstance(custom_code, bytes):
            custom_code = custom_code.decode('utf-8')
        return ('custom', custom_code)
    else:
        # Unknown case
        return ('unknown', None)


def merge_operator_codes_with_deduplication(model1, model2):
    """Merge operator codes from two models while removing duplicates.

    Args:
        model1: First Circle model
        model2: Second Circle model

    Returns:
        tuple: (merged_operator_codes, model2_to_merged_mapping)
    """
    # Start with first model's operator codes
    merged_operator_codes = list(model1.operatorCodes)

    # Create mapping table for operator codes
    # key: (builtinCode, customCode), value: new_index
    opcode_mapping = {}

    # Register first model's operator codes
    for i, op_code in enumerate(model1.operatorCodes):
        key = get_operator_code_key(op_code)
        opcode_mapping[key] = i

    # Process second model's operator codes (check for duplicates)
    model2_to_merged_mapping = {}  # model2's index → merged index

    for i, op_code in enumerate(model2.operatorCodes):
        key = get_operator_code_key(op_code)

        if key in opcode_mapping:
            # Duplicate operator code - use existing index
            model2_to_merged_mapping[i] = opcode_mapping[key]
        else:
            # New operator code - add it
            new_index = len(merged_operator_codes)
            merged_operator_codes.append(op_code)
            opcode_mapping[key] = new_index
            model2_to_merged_mapping[i] = new_index

    return merged_operator_codes, model2_to_merged_mapping


def create_tensor_map_list(subgraph, tensor_indices):
    """Convert tensor indices to TensorMap objects for SignatureDef.

    Args:
        subgraph: Subgraph containing the tensors
        tensor_indices: List of tensor indices

    Returns:
        list: List of TensorMapT objects
    """
    tensor_maps = []
    o2o.log(f"Creating tensor maps for {len(tensor_indices)} tensors")

    for i, tensor_idx in enumerate(tensor_indices):
        # Skip optional inputs (-1 indicates unused optional input)
        if tensor_idx == -1:
            continue

        # Ensure tensor index is valid
        if 0 <= tensor_idx < len(subgraph.tensors):
            tensor_map = circle.TensorMapT()

            # Get tensor name, use fallback if no name exists
            tensor_name = o2o.get_tensor_name(subgraph.tensors[tensor_idx])
            if not tensor_name:
                tensor_name = f"tensor_{tensor_idx}"

            # Encode name as UTF-8 bytes for FlatBuffers compatibility
            tensor_map.name = tensor_name.encode('utf-8')
            tensor_map.tensorIndex = int(tensor_idx)  # Convert numpy.int32 to int

            o2o.log(f"  TensorMap {i}: name='{tensor_name}', index={tensor_idx}")
            tensor_maps.append(tensor_map)
        else:
            o2o.log(f"Warning: Invalid tensor index {tensor_idx} in signature creation")

    return tensor_maps


def create_signatures(model, sig_name_0, sig_name_1):
    """Create signature definitions for the merged model.

    Args:
        model: Merged Circle model
        sig_name_0: Name for first subgraph signature
        sig_name_1: Name for second subgraph signature
    """
    if not hasattr(model, 'signatureDefs'):
        model.signatureDefs = []

    # Create signature for first subgraph
    sig0 = circle.SignatureDefT()
    sig0.subgraphIndex = 0
    sig0.signatureKey = sig_name_0.encode('utf-8')

    # Create TensorMap lists for inputs and outputs
    if model.subgraphs[0].inputs is not None and len(model.subgraphs[0].inputs) > 0:
        sig0.inputs = create_tensor_map_list(model.subgraphs[0], model.subgraphs[0].inputs)
    else:
        sig0.inputs = []

    if model.subgraphs[0].outputs is not None and len(model.subgraphs[0].outputs) > 0:
        sig0.outputs = create_tensor_map_list(model.subgraphs[0], model.subgraphs[0].outputs)
    else:
        sig0.outputs = []

    # Create signature for second subgraph
    sig1 = circle.SignatureDefT()
    sig1.subgraphIndex = 1
    sig1.signatureKey = sig_name_1.encode('utf-8')

    # Create TensorMap lists for inputs and outputs
    if model.subgraphs[1].inputs is not None and len(model.subgraphs[1].inputs) > 0:
        sig1.inputs = create_tensor_map_list(model.subgraphs[1], model.subgraphs[1].inputs)
    else:
        sig1.inputs = []

    if model.subgraphs[1].outputs is not None and len(model.subgraphs[1].outputs) > 0:
        sig1.outputs = create_tensor_map_list(model.subgraphs[1], model.subgraphs[1].outputs)
    else:
        sig1.outputs = []

    model.signatureDefs = [sig0, sig1]


def merge_models_with_signatures(model1, model2, sig_name_0, sig_name_1):
    """Merge two Circle models by keeping subgraphs separate and adding signatures.

    Args:
        model1: First Circle model
        model2: Second Circle model
        sig_name_0: Signature name for first subgraph
        sig_name_1: Signature name for second subgraph

    Returns:
        circle.ModelT: Merged model with signatures
    """
    # Validate that both models have exactly one subgraph
    if not model1.subgraphs or len(model1.subgraphs) != 1:
        o2o.log("Error: First model must have exactly one subgraph")
        sys.exit(1)

    if not model2.subgraphs or len(model2.subgraphs) != 1:
        o2o.log("Error: Second model must have exactly one subgraph")
        sys.exit(1)

    o2o.log(f"Merging models:")
    o2o.log(f"  Model 1: {len(model1.subgraphs[0].tensors)} tensors, {len(model1.subgraphs[0].operators)} operators")
    o2o.log(f"  Model 2: {len(model2.subgraphs[0].tensors)} tensors, {len(model2.subgraphs[0].operators)} operators")

    # Step 1: Merge buffers (simple append)
    merged_buffers = list(model1.buffers) + list(model2.buffers)
    buffer_offset = len(model1.buffers)

    # Step 2: Merge operator codes with deduplication
    merged_operator_codes, model2_opcode_mapping = merge_operator_codes_with_deduplication(model1, model2)

    # Step 3: Create merged subgraphs
    merged_subgraphs = []

    # First subgraph (keep as-is, no index remapping needed)
    subgraph0 = model1.subgraphs[0]
    merged_subgraphs.append(subgraph0)

    # Second subgraph (needs index remapping)
    subgraph1 = model2.subgraphs[0]

    # Remap buffer indices in second subgraph tensors
    for tensor in subgraph1.tensors:
        if tensor.buffer is not None and tensor.buffer != 0:
            tensor.buffer += buffer_offset

    # Remap operator code indices in second subgraph operators
    for operator in subgraph1.operators:
        if operator.opcodeIndex is not None:
            operator.opcodeIndex = model2_opcode_mapping[operator.opcodeIndex]

    merged_subgraphs.append(subgraph1)

    # Step 4: Create final merged model
    merged_model = circle.ModelT()
    merged_model.buffers = merged_buffers
    merged_model.operatorCodes = merged_operator_codes
    merged_model.subgraphs = merged_subgraphs

    # Step 5: Create signatures
    create_signatures(merged_model, sig_name_0, sig_name_1)

    o2o.log(f"Merge completed:")
    o2o.log(f"  Total buffers: {len(merged_buffers)}")
    o2o.log(f"  Total operator codes: {len(merged_operator_codes)}")
    o2o.log(f"  Total subgraphs: {len(merged_subgraphs)}")
    o2o.log(f"  Signatures: ['{sig_name_0}', '{sig_name_1}']")

    return merged_model


def main():
    """Main function to merge two Circle models with signatures."""
    parser = argparse.ArgumentParser(
        description='Merge two Circle models by appending subgraphs with signatures'
    )
    parser.add_argument('first_circle', help='First Circle model file')
    parser.add_argument('second_circle', help='Second Circle model file')
    parser.add_argument(
        '--sig-names',
        default='subgraph_0;subgraph_1',
        help='Signature names for subgraphs, separated by semicolon (e.g., "prefill;decode")'
    )
    args = parser.parse_args()

    # Parse signature names
    sig_names = args.sig_names.split(';')
    if len(sig_names) != 2:
        o2o.log("Error: --sig-names must contain exactly 2 names separated by semicolon")
        sys.exit(1)

    sig_name_0, sig_name_1 = sig_names[0].strip(), sig_names[1].strip()

    if not sig_name_0 or not sig_name_1:
        o2o.log("Error: Signature names cannot be empty")
        sys.exit(1)

    o2o.log(f"Loading models...")
    o2o.log(f"  First model: {args.first_circle}")
    o2o.log(f"  Second model: {args.second_circle}")
    o2o.log(f"  Signature names: ['{sig_name_0}', '{sig_name_1}']")

    # Load both models explicitly
    try:
        model0 = o2o.load_circle_model(args.first_circle)
        model1 = o2o.load_circle_model(args.second_circle)
    except Exception as e:
        o2o.log(f"Error loading models: {e}")
        sys.exit(1)

    # Merge models with signatures
    try:
        merged_model = merge_models_with_signatures(model0, model1, sig_name_0, sig_name_1)
    except Exception as e:
        o2o.log(f"Error merging models: {e}")
        sys.exit(1)

    # Output to stdout
    try:
        o2o.save_model_to_stdout(merged_model)
        o2o.log("Successfully saved merged model to stdout")
    except Exception as e:
        o2o.log(f"Error saving merged model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
