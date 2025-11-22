#!/usr/bin/env python3

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import circle
import o2o

# Import specific Circle types for better type annotations
from circle import (TensorT, OperatorT, SubGraphT, ModelT, BufferT, OperatorCodeT,
                    BuiltinOperator, TensorType)


def is_effectively_2d(shape):
    """Check if a tensor shape is effectively 2D (all leading dimensions are 1)"""
    if len(shape) < 2:
        return False  # Cannot be effectively 2D if less than 2 dimensions
    return all(dim == 1 for dim in shape[:-2])


def count_tensor_usage(model, tensor_index):
    """Count how many operators use a specific tensor as input"""
    count = 0
    for subgraph_idx, subgraph in enumerate(model.subgraphs):
        for operator in subgraph.operators:
            if operator.inputs is not None:
                for input_idx in operator.inputs:
                    if input_idx == tensor_index:
                        count += 1
    return count


def create_transpose_permutation_tensor(model: 'circle.ModelT',
                                        subgraph: 'circle.SubGraphT', rank: int) -> int:
    """Create a permutation tensor for transposing last two dimensions."""
    # Create permutation: [0, 1, ..., rank-3, rank-1, rank-2]
    perm_shape = [rank]
    perm_data = list(range(rank))
    perm_data[-1], perm_data[-2] = perm_data[-2], perm_data[-1]  # Swap last two

    # Create buffer for permutation data
    perm_buffer = circle.BufferT()
    perm_buffer.data = np.array(perm_data, dtype=np.int32).tobytes()
    model.buffers.append(perm_buffer)
    buffer_index = len(model.buffers)

    # Create tensor
    perm_tensor = circle.TensorT()
    perm_tensor.shape = perm_shape
    perm_tensor.type = circle.TensorType.INT32
    perm_tensor.buffer = buffer_index
    perm_tensor.name = f"transpose_perm_{len(subgraph.tensors)}"
    subgraph.tensors.append(perm_tensor)
    tensor_index = len(subgraph.tensors) - 1

    return tensor_index


def create_reshaped_tensor(original_tensor: 'circle.TensorT',
                           new_shape: List[int]) -> 'circle.TensorT':
    """Create a new tensor with the specified shape based on the original tensor"""
    new_tensor = circle.TensorT()
    new_tensor.shape = new_shape
    new_tensor.type = original_tensor.type
    new_tensor.buffer = original_tensor.buffer
    new_tensor.name = original_tensor.name
    new_tensor.quantization = original_tensor.quantization
    new_tensor.isVariable = original_tensor.isVariable
    new_tensor.sparsity = original_tensor.sparsity
    new_tensor.shapeSignature = original_tensor.shapeSignature
    new_tensor.hasRank = original_tensor.hasRank
    new_tensor.variantTensors = original_tensor.variantTensors
    new_tensor.compressionType = original_tensor.compressionType
    return new_tensor


def reshape_fc_weights(model: 'circle.ModelT', subgraph: 'circle.SubGraphT',
                       fc_op_idx: int) -> None:
    """Reshape FullyConnected weights from effectively 2D to 2D for a specific operator"""
    fc_op = subgraph.operators[fc_op_idx]

    # Get the weights tensor (typically the second input)
    if len(fc_op.inputs) < 2:
        return

    weights_index = fc_op.inputs[1]  # Weights is usually the second input
    weights_tensor = subgraph.tensors[weights_index]

    # Check if the weights tensor is effectively 2D
    if len(weights_tensor.shape) > 2 and is_effectively_2d(weights_tensor.shape):
        # Ensure keepNumDims is True
        if hasattr(fc_op.builtinOptions, 'keepNumDims'):
            fc_op.builtinOptions.keepNumDims = True

        # Check if this tensor is used by multiple operators
        usage_count = count_tensor_usage(model, weights_index)

        if usage_count > 1:
            # Create a new tensor for this operator to avoid affecting others
            new_shape = weights_tensor.shape[-2:]  # Remove leading dimensions of 1
            new_tensor = create_reshaped_tensor(weights_tensor, new_shape)

            # Add the new tensor to the subgraph
            new_tensor_index = len(subgraph.tensors)
            subgraph.tensors.append(new_tensor)

            # Update the operator input to use the new tensor
            fc_op.inputs[1] = new_tensor_index

            o2o.log(
                f"Created reshaped weight tensor {new_tensor_index} for FC operator at {fc_op_idx}"
            )
        else:
            # Directly modify the tensor shape since it's only used once
            original_shape = weights_tensor.shape
            weights_tensor.shape = weights_tensor.shape[-2:]
            o2o.log(
                f"Reshaped weight tensor {weights_index} from {original_shape} to {weights_tensor.shape}"
            )


def add_rhs_transpose_if_needed(model: 'circle.ModelT', subgraph: 'circle.SubGraphT',
                                bmm_op_idx: int, rhs_tensor_index: int,
                                rhs_tensor: 'circle.TensorT') -> int:
    """Add TRANSPOSE operator for RHS if K != 1 OR B != 1."""
    if len(rhs_tensor.shape) < 3:
        # Need at least 3 dimensions: [B, K, N]
        return rhs_tensor_index

    B = rhs_tensor.shape[0]
    K = rhs_tensor.shape[1]

    # Skip transpose if both B = 1 and K = 1
    if B == 1 and K == 1:
        o2o.log(
            f"RHS tensor shape {rhs_tensor.shape} has B=1 and K=1, skipping transpose")
        return rhs_tensor_index

    o2o.log(f"Adding transpose for RHS tensor shape {rhs_tensor.shape} (B={B}, K={K})")

    # Create permutation tensor
    rank = len(rhs_tensor.shape)
    perm_tensor_index = create_transpose_permutation_tensor(model, subgraph, rank)

    # Create output tensor for transposed RHS
    transposed_rhs_tensor = circle.TensorT()
    transposed_rhs_tensor.shape = list(rhs_tensor.shape)
    transposed_rhs_tensor.shape[-1], transposed_rhs_tensor.shape[
        -2] = transposed_rhs_tensor.shape[-2], transposed_rhs_tensor.shape[-1]
    transposed_rhs_tensor.type = rhs_tensor.type
    transposed_rhs_tensor.buffer = 0  # No buffer (intermediate tensor)
    transposed_rhs_tensor.name = f"transposed_rhs_{len(subgraph.tensors)}"
    subgraph.tensors.append(transposed_rhs_tensor)
    transposed_rhs_tensor_index = len(subgraph.tensors) - 1

    # Create TRANSPOSE operator
    transpose_op = circle.OperatorT()
    transpose_op.opcodeIndex = o2o.get_or_create_operator_code(
        model, circle.BuiltinOperator.TRANSPOSE)
    transpose_op.inputs = [rhs_tensor_index, perm_tensor_index]
    transpose_op.outputs = [transposed_rhs_tensor_index]
    transpose_op.builtinOptionsType = circle.BuiltinOptions.TransposeOptions
    transpose_options = circle.TransposeOptionsT()
    transpose_op.builtinOptions = transpose_options

    # Insert TRANSPOSE operator after BATCH_MATMUL
    subgraph.operators.insert(bmm_op_idx + 1, transpose_op)

    return transposed_rhs_tensor_index


def fuse_bmm_transpose() -> None:
    """Main function to add RHS transpose before fusing batchmatmul(lhs, rhs) to fullyconnected(transposed_rhs, lhs) when lhs is constant."""
    o2o.log("Loading model from stdin")
    model = o2o.load_model_from_stdin()

    if not model.subgraphs:
        o2o.log("Model has no subgraphs. Exiting.")
        o2o.save_model_to_stdout(model)  # Output to stdout for consistency
        return

    subgraph = model.subgraphs[0]  # Assuming single subgraph for now, can be extended
    tensors_to_potentially_remove = set()
    # Define operators to remove (empty list for now)
    operators_to_remove = []  # No operators to remove by default

    # Iterate backwards to safely remove operators
    for i in range(len(subgraph.operators) - 1, -1, -1):
        transpose_op = subgraph.operators[i]

        # Check if current operator is TRANSPOSE
        transpose_opcode = model.operatorCodes[transpose_op.opcodeIndex]
        if transpose_opcode.builtinCode != circle.BuiltinOperator.TRANSPOSE:
            continue

        if len(transpose_op.inputs) != 2:
            o2o.log(
                f"Transpose operator at index {i} has invalid number of inputs. Skipping."
            )
            continue

        transpose_input_tensor_idx = transpose_op.inputs[0]
        bmm_op_idx, bmm_op = o2o.find_operator_by_output(subgraph,
                                                         transpose_input_tensor_idx)

        # Check if the found operator is BATCH_MATMUL
        if bmm_op is None or model.operatorCodes[
                bmm_op.opcodeIndex].builtinCode != circle.BuiltinOperator.BATCH_MATMUL:
            continue

        lhs_tensor_index = bmm_op.inputs[0]
        rhs_tensor_index = bmm_op.inputs[1]

        lhs_tensor = o2o.get_tensor_by_index(subgraph, lhs_tensor_index)
        rhs_tensor = o2o.get_tensor_by_index(subgraph, rhs_tensor_index)

        if not lhs_tensor or not rhs_tensor:
            o2o.log(
                f"Could not find LHS or RHS tensor for BATCH_MATMUL at index {bmm_op_idx}. Skipping."
            )
            continue

        # Crucial check: LHS must be constant
        if not o2o.is_tensor_constant(lhs_tensor, model.buffers):
            o2o.log(
                f"LHS tensor '{lhs_tensor.name if lhs_tensor.name else lhs_tensor_index}' for BATCH_MATMUL at index {bmm_op_idx} is not constant. Skipping fusion."
            )
            continue

        # Verify Transpose permutation (assuming transpose of last two dims)
        # e.g. for [..., M, N] -> [..., N, M], permutation is [..., dim_N-1, dim_N-2]
        # For a 2D tensor [M, N] -> [N, M], permutation is [1, 0]
        # For a 3D tensor [B, M, N] -> [B, N, M], permutation is [0, 2, 1]
        valid_permutation = False
        perm_tensor_index = transpose_op.inputs[1]
        perm_tensor = o2o.get_tensor_by_index(subgraph, perm_tensor_index)

        if perm_tensor and o2o.is_tensor_constant(perm_tensor, model.buffers):
            # Get permutation data from buffer using the new helper function
            perm = o2o.from_buffer(perm_tensor.buffer, model.buffers)
            if perm is None:
                o2o.log(
                    f"Could not read permutation buffer at index {perm_tensor.buffer}")
                valid_permutation = False
            else:
                # Use the tensor's shape to determine the actual permutation length
                perm_length = perm_tensor.shape[0] if perm_tensor.shape else len(perm)
                perm = perm[:perm_length]  # Trim to actual length

                if len(perm) >= 2:  # At least 2D
                    # Check if the last two elements of permutation are swapped
                    # and other elements are in their original ascending order (0, 1, 2, ...)
                    expected_perm_prefix = list(range(len(perm) - 2))
                    actual_perm_prefix = perm[:-2]

                    if np.all(actual_perm_prefix == expected_perm_prefix) and \
                        perm[-2] == len(perm) - 1 and \
                        perm[-1] == len(perm) - 2:
                        valid_permutation = True
        else:
            o2o.log(
                f"Permutation tensor for TRANSPOSE at index {i} is not constant or not found. Skipping."
            )

        if not valid_permutation:
            o2o.log(
                f"TRANSPOSE operator at index {i} does not have a simple last-two-dim permutation. Skipping fusion."
            )
            continue

        # Add TRANSPOSE for RHS if needed (K != 1 OR B != 1)
        final_rhs_tensor_index = add_rhs_transpose_if_needed(model, subgraph, bmm_op_idx,
                                                             rhs_tensor_index, rhs_tensor)

        # Create the new FULLY_CONNECTED operator
        fc_op = circle.OperatorT()
        fc_op.opcodeIndex = o2o.get_or_create_operator_code(
            model, circle.BuiltinOperator.FULLY_CONNECTED)
        # Set inputs: [transposed_rhs, original_lhs, -1] where -1 means bias not exists
        fc_op.inputs = [final_rhs_tensor_index, lhs_tensor_index, -1]
        # Set outputs: same as the original TRANSPOSE operator
        fc_op.outputs = list(transpose_op.outputs)  # Make a copy

        # Configure FULLY_CONNECTED options
        fc_op.builtinOptionsType = (circle.BuiltinOptions.FullyConnectedOptions)
        fc_options = circle.FullyConnectedOptionsT()
        fc_options.keepNumDims = True  # Important to preserve batch dimensions from BATCH_MATMUL
        fc_op.builtinOptions = fc_options

        # Add the new operator to the subgraph
        # Insert it at the position of the original BATCH_MATMUL operator
        o2o.log(f"Replacing batchmatmul at {bmm_op_idx} with fullyconnected")
        subgraph.operators[bmm_op_idx] = fc_op

        # Reshape the weights of the newly created FC operator
        reshape_fc_weights(model, subgraph, bmm_op_idx)

        # Mark the original TRANSPOSE operator for removal
        operators_to_remove.append(i)

        # The tensor connecting BMM and Transpose (bmm_output_tensor_index) is now an intermediate
        # output of the new FC op. If it's not used by any other op, it could be cleaned up.
        # For now, we just mark it. Actual removal is more complex (needs usage check).
        tensors_to_potentially_remove.add(transpose_input_tensor_idx)

    # Remove operators marked for removal (iterate backwards again for safe removal)
    for i in sorted(list(operators_to_remove), reverse=True):
        if 0 <= i < len(subgraph.operators):
            o2o.log(f"Removing transpose operator at index {i}")
            del subgraph.operators[i]

    # Note: Cleanup of unused tensors and operator codes is a more advanced step
    # and not implemented here for simplicity, but would be part of a production-ready script.
    o2o.log(f"TODO: Remove tensors at {tensors_to_potentially_remove}")

    o2o.save_model_to_stdout(model)


if __name__ == "__main__":
    # Directly invoke processing; I/O handled via stdin/stdout
    fuse_bmm_transpose()
