#!/usr/bin/env python3

import os
import numpy as np
import circle
import o2o


def create_test_bmm_k_not_1_model(output_file):
    """Create a test Circle model with BATCH_MATMUL where RHS K != 1."""

    # Create model
    model = circle.ModelT()
    model.version = 3
    model.operatorCodes = []
    model.subgraphs = []
    model.buffers = []
    model.metadataBuffer = []

    # Create subgraph
    subgraph = circle.SubGraphT()
    subgraph.tensors = []
    subgraph.inputs = []
    subgraph.outputs = []
    subgraph.operators = []
    subgraph.name = "main"

    # Create buffers in CORRECT order (output buffer last, proper alignment)
    # B(0) - empty_sentinel_buffer (always existent empty buffer for tensors with no data buffer)
    empty_sentinel_buffer = circle.BufferT()
    # No data assignment for empty buffer (buffer.data remains None)
    model.buffers.append(empty_sentinel_buffer)

    # B(1) - BMM rhs : model input
    bmm_rhs_buffer = circle.BufferT()
    # No data assignment for input buffer (buffer.data remains None)
    model.buffers.append(bmm_rhs_buffer)

    # B(2) - BMM lhs : constant (16-byte aligned)
    bmm_lhs_data = np.random.rand(1, 4, 3).astype(np.float32)  # [B, M, K] = [1, 4, 3]
    bmm_lhs_buffer = circle.BufferT()
    # Align to 16 bytes as required by circle_schema.fbs Buffer.force_align: 16
    raw_data = bmm_lhs_data.tobytes()
    padded_data = raw_data + b'\x00' * (16 - len(raw_data) % 16) if len(
        raw_data) % 16 != 0 else raw_data
    bmm_lhs_buffer.data = padded_data
    model.buffers.append(bmm_lhs_buffer)

    # B(3) - Transpose perm : constant (16-byte aligned)
    perm_data = np.array(
        [0, 2, 1], dtype=np.int32)  # Transpose last two dims: [B, M, N] -> [B, N, M]
    perm_buffer = circle.BufferT()
    # Align to 16 bytes as required by circle_schema.fbs Buffer.force_align: 16
    raw_perm_data = perm_data.tobytes()
    padded_perm_data = raw_perm_data + b'\x00' * (16 - len(raw_perm_data) % 16) if len(
        raw_perm_data) % 16 != 0 else raw_perm_data
    perm_buffer.data = padded_perm_data
    model.buffers.append(perm_buffer)

    # B(4) - BMM output (intermediate, no data)
    bmm_output_buffer = circle.BufferT()
    # No data assignment for intermediate buffer (buffer.data remains None)
    model.buffers.append(bmm_output_buffer)

    # B(5) - TRANSPOSE output (final output, moved to last)
    transpose_output_buffer = circle.BufferT()
    # No data assignment for output buffer (buffer.data remains None)
    model.buffers.append(transpose_output_buffer)

    # Create RHS input tensor with K != 1 (K=3)
    bmm_rhs_tensor = circle.TensorT()
    bmm_rhs_tensor.shape = [1, 3, 5]  # [B, K, N] = [1, 3, 5], K=3 != 1
    bmm_rhs_tensor.type = circle.TensorType.FLOAT32
    bmm_rhs_tensor.buffer = 1  # B(1) - dedicated input buffer
    bmm_rhs_tensor.name = "bmm_rhs_input"
    subgraph.tensors.append(bmm_rhs_tensor)
    bmm_rhs_tensor_index = len(subgraph.tensors) - 1
    subgraph.inputs.append(bmm_rhs_tensor_index)  # Add to subgraph inputs

    # Create LHS constant tensor
    bmm_lhs_tensor = circle.TensorT()
    bmm_lhs_tensor.shape = [1, 4, 3]  # [B, M, K] = [1, 4, 3]
    bmm_lhs_tensor.type = circle.TensorType.FLOAT32
    bmm_lhs_tensor.buffer = 2  # B(2) - dedicated constant buffer with data
    bmm_lhs_tensor.name = "bmm_lhs_constant"
    subgraph.tensors.append(bmm_lhs_tensor)
    bmm_lhs_tensor_index = len(subgraph.tensors) - 1
    # Note: LHS is constant, so NOT added to subgraph.inputs

    # Create permutation tensor for TRANSPOSE
    perm_tensor = circle.TensorT()
    perm_tensor.shape = [3]
    perm_tensor.type = circle.TensorType.INT32
    perm_tensor.buffer = 3  # B(3) - dedicated constant buffer with data
    perm_tensor.name = "transpose_perm"
    subgraph.tensors.append(perm_tensor)
    perm_tensor_index = len(subgraph.tensors) - 1

    # Create BATCH_MATMUL output tensor
    bmm_output_tensor = circle.TensorT()
    bmm_output_tensor.shape = [1, 4, 5]  # [B, M, N] = [1, 4, 5]
    bmm_output_tensor.type = circle.TensorType.FLOAT32
    bmm_output_tensor.buffer = 4  # B(4) - intermediate buffer (no data)
    bmm_output_tensor.name = "bmm_output"
    subgraph.tensors.append(bmm_output_tensor)
    bmm_output_tensor_index = len(subgraph.tensors) - 1

    # Create final output tensor
    transpose_output_tensor = circle.TensorT()
    transpose_output_tensor.shape = [1, 5, 4]  # [B, N, M] = [1, 5, 4] after transpose
    transpose_output_tensor.type = circle.TensorType.FLOAT32
    transpose_output_tensor.buffer = 5  # B(5) - dedicated output buffer (last index)
    transpose_output_tensor.name = "transpose_output"
    subgraph.tensors.append(transpose_output_tensor)
    transpose_output_tensor_index = len(subgraph.tensors) - 1
    subgraph.outputs.append(transpose_output_tensor_index)

    # Create operator codes
    # BATCH_MATMUL
    bmm_opcode = circle.OperatorCodeT()
    bmm_opcode.builtinCode = circle.BuiltinOperator.BATCH_MATMUL
    bmm_opcode.deprecatedBuiltinCode = circle.BuiltinOperator.BATCH_MATMUL
    bmm_opcode.version = 1
    model.operatorCodes.append(bmm_opcode)
    bmm_opcode_index = len(model.operatorCodes) - 1

    # TRANSPOSE
    transpose_opcode = circle.OperatorCodeT()
    transpose_opcode.builtinCode = circle.BuiltinOperator.TRANSPOSE
    transpose_opcode.deprecatedBuiltinCode = circle.BuiltinOperator.TRANSPOSE
    transpose_opcode.version = 1
    model.operatorCodes.append(transpose_opcode)
    transpose_opcode_index = len(model.operatorCodes) - 1

    # Create BATCH_MATMUL operator
    bmm_op = circle.OperatorT()
    bmm_op.opcodeIndex = bmm_opcode_index
    bmm_op.inputs = [bmm_lhs_tensor_index,
                     bmm_rhs_tensor_index]  # LHS constant, RHS input
    bmm_op.outputs = [bmm_output_tensor_index]  # BMM output
    bmm_op.builtinOptionsType = circle.BuiltinOptions.BatchMatMulOptions
    bmm_options = circle.BatchMatMulOptionsT()
    bmm_options.adjointLhs = False  # Fixed: adjacentX -> adjointLhs
    bmm_options.adjointRhs = False  # Fixed: adjacentY -> adjointRhs
    bmm_options.asymmetricQuantizeInputs = False  # Added missing field
    bmm_options.fusedActivationFunction = circle.ActivationFunctionType.NONE
    bmm_op.builtinOptions = bmm_options
    subgraph.operators.append(bmm_op)

    # Create TRANSPOSE operator
    transpose_op = circle.OperatorT()
    transpose_op.opcodeIndex = transpose_opcode_index
    transpose_op.inputs = [bmm_output_tensor_index,
                           perm_tensor_index]  # BMM output, permutation
    transpose_op.outputs = [transpose_output_tensor_index]  # Final output
    transpose_op.builtinOptionsType = circle.BuiltinOptions.TransposeOptions
    transpose_options = circle.TransposeOptionsT()
    transpose_op.builtinOptions = transpose_options
    subgraph.operators.append(transpose_op)

    # Add subgraph to model
    model.subgraphs.append(subgraph)

    # Save model
    o2o.save_circle_model(model, output_file)
    o2o.log(f"Test model saved to {output_file}")
    o2o.log(f"Model structure:")
    o2o.log(f"  LHS constant tensor shape: {bmm_lhs_tensor.shape} (with actual data)")
    o2o.log(
        f"  RHS input tensor shape: {bmm_rhs_tensor.shape} (K={bmm_rhs_tensor.shape[1]} != 1)"
    )
    o2o.log(f"  BMM output tensor shape: {bmm_output_tensor.shape}")
    o2o.log(
        f"  TRANSPOSE output tensor shape: {transpose_output_tensor.shape} (after transpose)"
    )
    o2o.log(f"  Model inputs: {[subgraph.tensors[i].name for i in subgraph.inputs]}")
    o2o.log(f"  Model outputs: {[subgraph.tensors[i].name for i in subgraph.outputs]}")
    o2o.log(
        f"  Operations: BATCH_MATMUL(LHS_constant + RHS_input) -> TRANSPOSE(properly connected)"
    )


if __name__ == "__main__":
    # Generate output filename from current script filename
    # e.g., cvt.bmm_lhs_const.fc.circle_gen.py -> cvt.bmm_lhs_const.fc.circle
    script_name = os.path.basename(__file__)
    output_file = script_name.replace('gen_circle.', '').replace('.py', '.circle')

    create_test_bmm_k_not_1_model(output_file)
