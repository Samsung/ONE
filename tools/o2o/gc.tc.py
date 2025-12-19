#!/usr/bin/env python3
import circle
import o2o
import sys


def generate_test_model():
    model = circle.ModelT()
    subgraph = circle.SubGraphT()
    model.subgraphs = [subgraph]

    # Create tensors
    # T0: Input, used by Op
    # T1: Input, UNUSED
    # T2: Output, produced by Op
    # T3: Output, UNUSED (not produced by any Op)

    t0 = circle.TensorT()
    t0.name = "input_used"
    t0.shape = [1, 2]
    t0.type = circle.TensorType.FLOAT32

    t1 = circle.TensorT()
    t1.name = "input_unused"
    t1.shape = [1, 2]
    t1.type = circle.TensorType.FLOAT32

    t2 = circle.TensorT()
    t2.name = "output_used"
    t2.shape = [1, 2]
    t2.type = circle.TensorType.FLOAT32

    t3 = circle.TensorT()
    t3.name = "output_unused"
    t3.shape = [1, 2]
    t3.type = circle.TensorType.FLOAT32

    subgraph.tensors = [t0, t1, t2, t3]

    # Set inputs and outputs
    subgraph.inputs = [0, 1]  # t0, t1
    subgraph.outputs = [2, 3]  # t2, t3

    # Create an operator that uses T0 and produces T2
    # Use NEG (unary)

    neg_op = circle.OperatorT()
    neg_op.opcodeIndex = 0  # Will set code below
    neg_op.inputs = [0]  # Uses T0
    neg_op.outputs = [2]  # Produces T2

    subgraph.operators = [neg_op]

    # Add OperatorCode
    op_code = circle.OperatorCodeT()
    op_code.builtinCode = circle.BuiltinOperator.NEG
    op_code.version = 1
    model.operatorCodes = [op_code]

    # Add default empty buffer (Buffer 0)
    b0 = circle.BufferT()
    model.buffers = [b0]

    # Save to stdout
    o2o.save_model_to_stdout(model)


if __name__ == "__main__":
    generate_test_model()
