#!/usr/bin/env python3

import numpy as np
import circle
import o2o


def find_operator_by_output(subgraph, output_tensor_index):
    """Find the first operator that produces the given output tensor index."""
    for op_idx, operator in enumerate(subgraph.operators):
        if operator.outputs and output_tensor_index in operator.outputs:
            return op_idx, operator
    return None, None


def find_attention_blocks(model, subgraph):
    """Find all attention blocks in the subgraph."""
    attention_blocks = []

    # Pattern: 45 operators per attention block
    # First block: operators 19-63
    # Second block: operators 84-128
    # Third block: operators 149-193
    # Fourth block: operators 194-238
    # Fifth block: operators 239-283
    # Sixth block: operators 284-328
    # Seventh block: operators 329-373
    # Eighth block: operators 374-418

    for layer_idx in range(8):
        start_op = 20 + (layer_idx * 64)  # 64 operators between blocks, starting from 20
        end_op = start_op + 43  # 44 operators total (20-63 inclusive)

        if end_op < len(subgraph.operators):
            # Verify this is an attention block by checking key operators
            # Should start with FULLY_CONNECTED and end with FULLY_CONNECTED
            first_op = subgraph.operators[start_op]
            last_op = subgraph.operators[end_op]

            first_opcode = model.operatorCodes[first_op.opcodeIndex]
            last_opcode = model.operatorCodes[last_op.opcodeIndex]

            if (first_opcode.builtinCode == circle.BuiltinOperator.FULLY_CONNECTED and
                    last_opcode.builtinCode == circle.BuiltinOperator.FULLY_CONNECTED):

                attention_blocks.append({
                    'layer_idx':
                    layer_idx,
                    'start_op':
                    start_op,
                    'end_op':
                    end_op,
                    'operators':
                    subgraph.operators[start_op:end_op + 1]
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

    return attention_blocks


def map_attention_inputs(subgraph, block, model):
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
        subgraph, "transformers.models.llama.modeling_llama.LlamaForCausalLM::mul_1")
    if position_cos_idx == -1:
        o2o.log("Could not find position_cos tensor")
        return None

    # 7. position_sin
    position_sin_idx = o2o.get_tensor_index_by_name(
        subgraph, "transformers.models.llama.modeling_llama.LlamaForCausalLM::mul_2")
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
    o2o.log("Loading model from stdin")
    model = o2o.load_model_from_stdin()

    if not model.subgraphs:
        o2o.log("Model has no subgraphs. Exiting.")
        return

    subgraph = model.subgraphs[0]  # Assuming single subgraph for now
    attention_blocks = find_attention_blocks(model, subgraph)

    o2o.log(f"Found {len(attention_blocks)} attention blocks to fuse")

    operators_to_remove = []

    for block in attention_blocks:
        # Map input tensors
        input_indices = map_attention_inputs(subgraph, block, model)
        if input_indices is None:
            o2o.log(
                f"Skipping attention block {block['layer_idx']} due to missing inputs")
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

    o2o.log(f"Removed {len(operators_to_remove)} intermediate operators")
    o2o.log(f"Model now has {len(subgraph.operators)} operators")

    o2o.save_model_to_stdout(model)


if __name__ == "__main__":
    # Directly invoke processing; I/O handled via stdin/stdout
    fuse_attention()
