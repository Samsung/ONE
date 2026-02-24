#!/usr/bin/env python3

import numpy as np
import sys
from typing import List, Optional, Tuple, Dict, Any
import circle
import o2o

# Import specific Circle types for better type annotations
from circle import (TensorT, OperatorT, SubGraphT, ModelT, BufferT, OperatorCodeT,
                    BuiltinOperator, TensorType)

# Map builtin codes to names for readability (for debug mode)
BUILTIN_NAMES = {
    v: k
    for k, v in circle.BuiltinOperator.__dict__.items() if not k.startswith('_')
}

# ============================================================================
# DEBUG FUNCTIONS
# ============================================================================


def inspect_ops(subgraph: 'circle.SubGraphT', model: 'circle.ModelT', limit: int = 100):
    """Inspect and print operator information (for debugging)."""
    print(f"{'Index':<5}  {'OpCode':27} {'BuiltinCode':^12} {'Weight Name':^55}")
    print("-" * 100)

    for i in range(min(limit, len(subgraph.operators))):
        op = subgraph.operators[i]
        opcode = model.operatorCodes[op.opcodeIndex]
        builtin_code = opcode.builtinCode
        name = BUILTIN_NAMES.get(builtin_code, str(builtin_code))

        extra_info = ""
        if builtin_code == circle.BuiltinOperator.FULLY_CONNECTED:
            # FC inputs: input, weights, bias (optional)
            if len(op.inputs) > 1:
                weight_tensor = o2o.get_tensor_by_index(subgraph, op.inputs[1])
                if weight_tensor:
                    extra_info = o2o.get_tensor_name(weight_tensor) or ""

        print(f"{i:>4}   {name:<27} {builtin_code:>5} {extra_info:>56}")


def extract_pattern(subgraph: 'circle.SubGraphT', model: 'circle.ModelT'):
    """Extract attention block pattern for debugging."""
    start_op = -1
    end_op = -1

    print("\nSearching for attention block pattern based on weight names...")

    for i, op in enumerate(subgraph.operators):
        opcode = model.operatorCodes[op.opcodeIndex]
        if opcode.builtinCode == circle.BuiltinOperator.FULLY_CONNECTED:
            if len(op.inputs) > 1:
                weight_tensor = o2o.get_tensor_by_index(subgraph, op.inputs[1])
                if weight_tensor:
                    weight_name = o2o.get_tensor_name(weight_tensor) or ""

                    # Look for start: attn_q_proj
                    if start_op == -1 and "attn_q_proj" in weight_name:
                        start_op = i
                        print(f"Found start_op at {i} (Weight: {weight_name})")

                    # Look for end: attn_o_proj (must be after start)
                    if start_op != -1 and "attn_o_proj" in weight_name:
                        end_op = i
                        print(f"Found end_op at {i} (Weight: {weight_name})")
                        break

    if start_op != -1 and end_op != -1:
        pattern_codes = []
        for i in range(start_op, end_op + 1):
            op = subgraph.operators[i]
            opcode = model.operatorCodes[op.opcodeIndex]
            pattern_codes.append(opcode.builtinCode)

        print(f"\nExtracted range: {start_op} - {end_op}")
        print("ATTENTION_PATTERN_CODES = " + str(pattern_codes))

        if pattern_codes[0] == circle.BuiltinOperator.FULLY_CONNECTED:
            print("Verified: Pattern starts with FULLY_CONNECTED")
        else:
            print(f"Warning: Pattern starts with {pattern_codes[0]}")

        if pattern_codes[-1] == circle.BuiltinOperator.FULLY_CONNECTED:
            print("Verified: Pattern ends with FULLY_CONNECTED")
        else:
            print(f"Warning: Pattern ends with {pattern_codes[-1]}")

    else:
        print("Could not find attention block pattern using weight names.")


# ============================================================================
# ATTENTION FUSION FUNCTIONS
# ============================================================================


def find_attention_pattern(
        model: 'circle.ModelT',
        subgraph: 'circle.SubGraphT') -> Tuple[int, int, int, int, List[int]]:
    """
    Dynamically find attention block pattern parameters.

    Returns:
        Tuple of (start_offset, block_length, stride, num_blocks, pattern_codes)

    Raises:
        RuntimeError if pattern cannot be detected
    """
    first_block_start = -1
    first_block_end = -1
    second_block_start = -1

    # Find first and second attention blocks by searching for q_proj and o_proj
    for i, op in enumerate(subgraph.operators):
        opcode = model.operatorCodes[op.opcodeIndex]
        if opcode.builtinCode == circle.BuiltinOperator.FULLY_CONNECTED:
            if len(op.inputs) > 1:
                weight_tensor = o2o.get_tensor_by_index(subgraph, op.inputs[1])
                if weight_tensor:
                    weight_name = o2o.get_tensor_name(weight_tensor) or ""

                    # Look for first block start: first attn_q_proj
                    if first_block_start == -1 and "attn_q_proj" in weight_name:
                        first_block_start = i
                        o2o.log(f"Detected first attention block start at {i}")

                    # Look for first block end: first attn_o_proj (after start)
                    elif first_block_start != -1 and first_block_end == -1 and "attn_o_proj" in weight_name:
                        first_block_end = i
                        o2o.log(f"Detected first attention block end at {i}")

                    # Look for second block start: second attn_q_proj (after first block end)
                    elif first_block_end != -1 and second_block_start == -1 and "attn_q_proj" in weight_name:
                        second_block_start = i
                        o2o.log(f"Detected second attention block start at {i}")
                        break

    if first_block_start == -1 or first_block_end == -1 or second_block_start == -1:
        raise RuntimeError(
            "Could not detect attention pattern dynamically. Unable to find first and second attention blocks."
        )

    block_length = first_block_end - first_block_start
    stride = second_block_start - first_block_start

    # Calculate number of blocks
    num_blocks = 0
    test_start = first_block_start
    while test_start + block_length < len(subgraph.operators):
        num_blocks += 1
        test_start += stride

    # Extract pattern codes from the first block
    pattern_codes = []
    for i in range(first_block_start, first_block_end + 1):
        op = subgraph.operators[i]
        opcode = model.operatorCodes[op.opcodeIndex]
        pattern_codes.append(opcode.builtinCode)

    o2o.log(
        f"Detected pattern: start={first_block_start}, block_length={block_length}, stride={stride}, num_blocks={num_blocks}"
    )
    return (first_block_start, block_length, stride, num_blocks, pattern_codes)


def find_attention_blocks(model: 'circle.ModelT',
                          subgraph: 'circle.SubGraphT') -> List[Dict[str, Any]]:
    """Find all attention blocks in the subgraph."""
    attention_blocks = []

    start_offset, block_length, stride, num_blocks, pattern_codes = find_attention_pattern(
        model, subgraph)

    for layer_idx in range(num_blocks):
        start_op = start_offset + (layer_idx * stride)
        end_op = start_op + block_length

        if end_op < len(subgraph.operators):
            # Verify this is an attention block by checking key operators
            # Should start with FULLY_CONNECTED and end with FULLY_CONNECTED
            first_op = subgraph.operators[start_op]
            last_op = subgraph.operators[end_op]

            first_opcode = model.operatorCodes[first_op.opcodeIndex]
            last_opcode = model.operatorCodes[last_op.opcodeIndex]

            if (first_opcode.builtinCode == circle.BuiltinOperator.FULLY_CONNECTED and
                    last_opcode.builtinCode == circle.BuiltinOperator.FULLY_CONNECTED):

                # Verify the entire block pattern matches the first block's pattern
                current_block_codes = []
                for i in range(start_op, end_op + 1):
                    op = subgraph.operators[i]
                    opcode = model.operatorCodes[op.opcodeIndex]
                    current_block_codes.append(opcode.builtinCode)

                if current_block_codes != pattern_codes:
                    o2o.log(
                        f"Pattern mismatch for block {layer_idx} at {start_op}: Opcode sequence does not match the first block."
                    )
                    continue

                attention_blocks.append({
                    'layer_idx':
                    layer_idx,
                    'start_op':
                    start_op,
                    'end_op':
                    end_op,
                    'operators':
                    subgraph.operators[start_op:end_op + 1],
                    'q_proj_op':
                    start_op,  # First FC is Q projection
                    'k_proj_op':
                    start_op + 3,  # K projection (after reshape, transpose)
                    'v_proj_op':
                    start_op + 6,  # V projection
                    'o_proj_op':
                    end_op  # Last FC is output projection
                })

                o2o.log(
                    f"Found attention block {layer_idx}: operators {start_op}-{end_op}")
            else:
                o2o.log(
                    f"Pattern mismatch for block starting at {start_op}: expected FULLY_CONNECTED->FULLY_CONNECTED, got {first_opcode.builtinCode}->{last_opcode.builtinCode}"
                )
        else:
            o2o.log(
                f"Block {layer_idx} exceeds operator count (start_op={start_op}, total={len(subgraph.operators)})"
            )

    o2o.log(f"Found {len(attention_blocks)} attention blocks to fuse")
    return attention_blocks


def map_attention_inputs(subgraph: 'circle.SubGraphT', block: Dict[str, Any],
                         model: 'circle.ModelT') -> Optional[List[int]]:
    """Map the 11 input tensors for attention fusion."""
    layer_idx = block['layer_idx']

    # 1. hidden_states (RMSNorm output)
    hidden_states_tensor = o2o.get_tensor_by_index(subgraph,
                                                   block['operators'][0].inputs[0])
    if not hidden_states_tensor:
        o2o.log(f"Could not find hidden_states tensor for layer {layer_idx}")
        return None

    # 2. wq (query weight)
    wq_name = f"tico::p_model_layers_{layer_idx}_self_attn_q_proj_weight"
    wq_idx = o2o.get_tensor_index_by_name(subgraph, wq_name)
    if wq_idx == -1:
        o2o.log(f"Could not find wq tensor: {wq_name}")
        return None

    # 3. wk (key weight)
    wk_name = f"tico::p_model_layers_{layer_idx}_self_attn_k_proj_weight"
    wk_idx = o2o.get_tensor_index_by_name(subgraph, wk_name)
    if wk_idx == -1:
        o2o.log(f"Could not find wk tensor: {wk_name}")
        return None

    # 4. wv (value weight)
    wv_name = f"tico::p_model_layers_{layer_idx}_self_attn_v_proj_weight"
    wv_idx = o2o.get_tensor_index_by_name(subgraph, wv_name)
    if wv_idx == -1:
        o2o.log(f"Could not find wv tensor: {wv_name}")
        return None

    # 5. wo (output weight)
    wo_name = f"tico::p_model_layers_{layer_idx}_self_attn_o_proj_weight"
    wo_idx = o2o.get_tensor_index_by_name(subgraph, wo_name)
    if wo_idx == -1:
        o2o.log(f"Could not find wo tensor: {wo_name}")
        return None

    # 6. position_cos
    position_cos_idx = o2o.get_tensor_index_by_name(
        subgraph, "transformers.models.llama.modeling_llama.LlamaForCausalLM::cos")
    if position_cos_idx == -1:
        o2o.log("Could not find position_cos tensor")
        return None

    # 7. position_sin
    position_sin_idx = o2o.get_tensor_index_by_name(
        subgraph, "transformers.models.llama.modeling_llama.LlamaForCausalLM::sin")
    if position_sin_idx == -1:
        o2o.log("Could not find position_sin tensor")
        return None

    # 8. attention_mask
    attention_mask_idx = o2o.get_tensor_index_by_name(
        subgraph, "transformers.models.llama.modeling_llama.LlamaModel::mul")
    if attention_mask_idx == -1:
        o2o.log("Could not find attention_mask tensor")
        return None

    # 9. past_key
    past_key_name = f"tico::past_key_values_key_cache_{layer_idx}"
    past_key_idx = o2o.get_tensor_index_by_name(subgraph, past_key_name)
    if past_key_idx == -1:
        o2o.log(f"Could not find past_key tensor: {past_key_name}")
        return None

    # 10. past_value
    past_value_name = f"tico::past_key_values_value_cache_{layer_idx}"
    past_value_idx = o2o.get_tensor_index_by_name(subgraph, past_value_name)
    if past_value_idx == -1:
        o2o.log(f"Could not find past_value tensor: {past_value_name}")
        return None

    # 11. cache_position
    cache_position_idx = o2o.get_tensor_index_by_name(subgraph, "tico::cache_position")
    if cache_position_idx == -1:
        o2o.log("Could not find cache_position tensor")
        return None

    # Find the tensor index for hidden_states (not buffer index)
    hidden_states_idx = o2o.get_tensor_index_by_name(
        subgraph,
        o2o.get_tensor_name(hidden_states_tensor)) if hidden_states_tensor.name else -1
    if hidden_states_idx == -1:
        o2o.log(f"Could not find tensor index for hidden_states in layer {layer_idx}")
        return None

    return [
        hidden_states_idx,  # hidden_states (tensor index, not buffer index)
        wq_idx,  # wq
        wk_idx,  # wk
        wv_idx,  # wv
        wo_idx,  # wo
        position_cos_idx,  # position_cos
        position_sin_idx,  # position_sin
        attention_mask_idx,  # attention_mask
        past_key_idx,  # past_key
        past_value_idx,  # past_value
        cache_position_idx  # cache_position
    ]


def fuse_attention():
    """Main function to fuse attention operators."""
    model = o2o.load_model_from_stdin()

    if not model.subgraphs:
        o2o.log("Model has no subgraphs. Exiting.")
        return

    # Process all subgraphs
    for subgraph_idx, subgraph in enumerate(model.subgraphs):
        o2o.log(f"Processing subgraph {subgraph_idx}...")

        attention_blocks = find_attention_blocks(model, subgraph)

        o2o.log(
            f"Found {len(attention_blocks)} attention blocks to fuse in subgraph {subgraph_idx}"
        )

        operators_to_remove = []

        for block in attention_blocks:
            # Map input tensors
            input_indices = map_attention_inputs(subgraph, block, model)
            if input_indices is None:
                o2o.log(
                    f"Skipping attention block {block['layer_idx']} due to missing inputs"
                )
                continue

            # Create ATTENTION operator
            attention_op = circle.OperatorT()
            attention_op.opcodeIndex = o2o.get_or_create_operator_code(
                model, circle.BuiltinOperator.ATTENTION)
            attention_op.inputs = input_indices
            attention_op.outputs = [block['operators'][-1].outputs[0]
                                    ]  # Use last operator's output

            # Configure AttentionOptions (empty since it's deprecated)
            attention_op.builtinOptionsType = circle.BuiltinOptions.AttentionOptions
            attention_op.builtinOptions = circle.AttentionOptionsT()

            # Replace the first operator with ATTENTION operator
            start_idx = block['start_op']
            subgraph.operators[start_idx] = attention_op

            # Mark intermediate operators for removal (except the first one which we replaced)
            for i in range(block['end_op'], start_idx, -1):
                operators_to_remove.append(i)

            o2o.log(
                f"Fused attention block {block['layer_idx']}: operators {block['start_op']}-{block['end_op']} -> ATTENTION"
            )

        # Remove marked operators in reverse order to avoid index shifting
        for i in sorted(operators_to_remove, reverse=True):
            if 0 <= i < len(subgraph.operators):
                del subgraph.operators[i]

        o2o.log(
            f"Removed {len(operators_to_remove)} intermediate operators from subgraph {subgraph_idx}"
        )
        o2o.log(f"Subgraph {subgraph_idx} now has {len(subgraph.operators)} operators")

    o2o.save_model_to_stdout(model)


if __name__ == "__main__":
    # Check for inspect mode flag
    if len(sys.argv) > 1 and sys.argv[1] == "--inspect":
        # Inspect mode: analyze operators and extract pattern (no fusion)
        o2o.log("Running in INSPECT mode")
        model = o2o.load_model_from_stdin()

        # Process all subgraphs in inspect mode
        for subgraph_idx, subgraph in enumerate(model.subgraphs):
            o2o.log(f"\n{'='*100}")
            o2o.log(f"Subgraph {subgraph_idx}")
            o2o.log(f"{'='*100}")

            # Inspect first 100 ops
            inspect_ops(subgraph, model)

            # Extract pattern dynamically
            extract_pattern(subgraph, model)
    else:
        # Normal mode: fuse attention blocks
        fuse_attention()
