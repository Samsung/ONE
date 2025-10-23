#!/usr/bin/env python3

import os
import numpy as np
import circle
import o2o

# Circle Model Buffer Usage Rules (based on circle_schema.fbs and analysis)
# ======================================================================
#
# Buffer Index Allocation Rules:
# - B(0): Always empty placeholder buffer (sentinel, must exist)
# - B(1+): Dedicated buffers for specific tensors
#
# Tensor-Buffer Assignment Rules:
# 1. Model Input Tensors:
#    - Get dedicated buffer index (e.g., B(1), B(2), ...)
#    - Buffer data is EMPTY (b'')
#    - Added to subgraph.inputs array
#    - Example: input tensor -> buffer index 1 (empty)
#
# 2. Model Output Tensors:
#    - Get dedicated buffer index (e.g., B(1), B(2), ...)
#    - Buffer data is EMPTY (b'')
#    - Added to subgraph.outputs array
#    - Example: output tensor -> buffer index 2 (empty)
#
# 3. Constant Tensors:
#    - Get dedicated buffer index (e.g., B(3), B(4), ...)
#    - Buffer data contains ACTUAL DATA (numpy.tobytes())
#    - NOT added to subgraph.inputs (internal to model)
#    - Example: constant tensor -> buffer index 3 (with data)
#
# 4. Intermediate Tensors:
#    - Get dedicated buffer index (e.g., B(4), B(5), ...)
#    - Buffer data is EMPTY (b'')
#    - NOT added to subgraph.inputs/outputs
#    - Example: intermediate result -> buffer index 4 (empty)
#    - IMPORTANT: Intermediate tensors are NOT constants, so they need dedicated buffers!
#
# Buffer Creation Order (Recommended):
# 1. B(0): Empty placeholder buffer (always first)
# 2. Input tensor buffers (empty data)
# 3. Output tensor buffers (empty data)
# 4. Constant tensor buffers (with actual data)
#
# Key Principles:
# - Each tensor type has specific buffer requirements
# - Model inputs/outputs MUST have dedicated buffers (even if empty)
# - Constants MUST have dedicated buffers with actual data
# - Intermediate results use buffer index 0
# - Buffer index assignment follows creation order in model.buffers array
#
# Reference: circle_schema.fbs - "buffer:uint" field documentation


def create_simple_add_model(output_file):
    """Create a simple Circle model with one ADD operator (similar to add.circle)."""

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

    # Create buffers in CORRECT order (output buffer last)
    # B(0) - empty_sentinel_buffer (always existent empty buffer for tensors with no data buffer)
    empty_sentinel_buffer = circle.BufferT()
    # No data assignment for empty buffer (buffer.data remains None)
    model.buffers.append(empty_sentinel_buffer)

    # B(1) - input tensor buffer (no data)
    input_buffer = circle.BufferT()
    # No data assignment for input buffer (buffer.data remains None)
    model.buffers.append(input_buffer)

    # B(2) - constant tensor buffer (with data, 16-byte aligned)
    const_data = np.array([1], dtype=np.int32)  # Simple constant value
    const_buffer = circle.BufferT()
    # Align to 16 bytes as required by circle_schema.fbs Buffer.force_align: 16
    raw_data = const_data.tobytes()
    padded_data = raw_data + b'\x00' * (16 - len(raw_data))  # 4 + 12 = 16 bytes
    const_buffer.data = padded_data
    model.buffers.append(const_buffer)

    # B(3) - output tensor buffer (no data) - MOVED TO LAST
    output_buffer = circle.BufferT()
    # No data assignment for output buffer (buffer.data remains None)
    model.buffers.append(output_buffer)

    # Create input tensor (ifm) - using dedicated buffer B(1)
    input_tensor = circle.TensorT()
    input_tensor.shape = [1, 1, 16]  # Same as add.circle
    input_tensor.type = circle.TensorType.INT32  # Using INT32
    input_tensor.buffer = 1  # B(1) - dedicated input buffer
    input_tensor.name = "ifm"
    subgraph.tensors.append(input_tensor)
    input_tensor_index = len(subgraph.tensors) - 1
    subgraph.inputs.append(input_tensor_index)

    # Create constant tensor (add_const) - using dedicated buffer B(2)
    const_tensor = circle.TensorT()
    const_tensor.shape = [1, 1, 1]  # Same as add.circle
    const_tensor.type = circle.TensorType.INT32
    const_tensor.buffer = 2  # B(2) - dedicated constant buffer with data
    const_tensor.name = "add_const"
    subgraph.tensors.append(const_tensor)
    const_tensor_index = len(subgraph.tensors) - 1

    # Create output tensor (ofm) - using dedicated buffer B(3) - MOVED TO LAST
    output_tensor = circle.TensorT()
    output_tensor.shape = [1, 1, 16]  # Same as add.circle
    output_tensor.type = circle.TensorType.INT32
    output_tensor.buffer = 3  # B(3) - dedicated output buffer (last index)
    output_tensor.name = "ofm"
    subgraph.tensors.append(output_tensor)
    output_tensor_index = len(subgraph.tensors) - 1
    subgraph.outputs.append(output_tensor_index)

    # Create ADD operator code
    add_opcode = circle.OperatorCodeT()
    add_opcode.builtinCode = circle.BuiltinOperator.ADD
    add_opcode.deprecatedBuiltinCode = circle.BuiltinOperator.ADD  # Fix: deprecatedBuiltinCode must be set to same as builtinCode
    add_opcode.version = 1
    model.operatorCodes.append(add_opcode)
    add_opcode_index = len(model.operatorCodes) - 1

    # Create ADD operator
    add_op = circle.OperatorT()
    add_op.opcodeIndex = add_opcode_index
    add_op.inputs = [input_tensor_index, const_tensor_index]  # ifm + add_const
    add_op.outputs = [output_tensor_index]  # = ofm
    add_op.builtinOptionsType = circle.BuiltinOptions.AddOptions
    add_options = circle.AddOptionsT()
    add_options.fusedActivationFunction = circle.ActivationFunctionType.NONE
    add_op.builtinOptions = add_options
    subgraph.operators.append(add_op)

    # Add subgraph to model
    model.subgraphs.append(subgraph)

    # Save model
    o2o.save_circle_model(model, output_file)
    o2o.log(f"Simple ADD model saved to {output_file}")
    o2o.log(f"Model structure:")
    o2o.log(f"  Input tensor: {input_tensor.name} shape={input_tensor.shape}")
    o2o.log(f"  Constant tensor: {const_tensor.name} shape={const_tensor.shape}")
    o2o.log(f"  Output tensor: {output_tensor.name} shape={output_tensor.shape}")
    o2o.log(f"  Operator: ADD")
    o2o.log(f"  Subgraph inputs: {[subgraph.tensors[i].name for i in subgraph.inputs]}")
    o2o.log(f"  Subgraph outputs: {[subgraph.tensors[i].name for i in subgraph.outputs]}")


if __name__ == "__main__":
    # Generate output filename from current script filename
    # e.g., add.gen_circle.py -> add.circle
    script_name = os.path.basename(__file__)
    output_file = script_name.replace('.gen_circle.py', '.circle')

    create_simple_add_model(output_file)
