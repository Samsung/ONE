#!/usr/bin/env python3

import numpy as np
import circle
import o2o


def is_effectively_2d(shape):
    """Check if a tensor shape is effectively 2D (all leading dimensions are 1)"""
    return all(dim == 1 for dim in shape[:-2])


def count_tensor_usage(model, tensor_index):
    """Count how many operators use a specific tensor as input"""
    count = 0
    for subgraph in model.subgraphs:
        for operator in subgraph.operators:
            if operator.inputs is not None:
                for input_idx in operator.inputs:
                    if input_idx == tensor_index:
                        count += 1
    return count


def create_new_tensor(original_tensor, new_shape):
    """Create a new tensor with the specified shape based on the original tensor"""
    new_tensor = circle.TensorT()
    new_tensor.shape = new_shape
    new_tensor.type = original_tensor.type
    new_tensor.buffer = original_tensor.buffer
    new_tensor.name = original_tensor.name + "_reshaped" if original_tensor.name else None
    new_tensor.quantization = original_tensor.quantization
    new_tensor.isVariable = original_tensor.isVariable
    new_tensor.sparsity = original_tensor.sparsity
    new_tensor.shapeSignature = original_tensor.shapeSignature
    new_tensor.hasRank = original_tensor.hasRank
    new_tensor.variantTensors = original_tensor.variantTensors
    new_tensor.compressionType = original_tensor.compressionType
    return new_tensor


def modify_fully_connected_weights():
    """Main function to modify FullyConnected weights from effectively 2D to 2D"""
    # Load the model using utility function
    model = o2o.load_model_from_stdin()

    # Process each subgraph
    for subgraph in model.subgraphs:
        # Create a mapping from old tensor indices to new tensor indices
        tensor_mapping = {}

        # First pass: identify and create new tensors for modification
        for i, operator in enumerate(subgraph.operators):
            # Check if this is a FullyConnected operator
            opcode = model.operatorCodes[operator.opcodeIndex]
            if opcode.builtinCode == circle.BuiltinOperator.FULLY_CONNECTED:
                # Get the weights tensor (typically the second input)
                if len(operator.inputs) >= 2:
                    weights_index = operator.inputs[
                        1]  # Weights is usually the second input
                    weights_tensor = subgraph.tensors[weights_index]

                    # Check if the weights tensor is effectively 2D
                    if len(weights_tensor.shape) > 2 and is_effectively_2d(
                            weights_tensor.shape):
                        operator.builtinOptions.keepNumDims = True
                        # Check if this tensor is used by multiple operators
                        usage_count = count_tensor_usage(model, weights_index)

                        if usage_count > 1:
                            # Create a new tensor for this operator to avoid affecting others
                            new_shape = weights_tensor.shape[
                                -2:]  # Remove leading dimensions of 1
                            new_tensor = create_new_tensor(weights_tensor, new_shape)

                            # Add the new tensor to the subgraph
                            new_tensor_index = len(subgraph.tensors)
                            subgraph.tensors.append(new_tensor)

                            # Update the mapping for this specific operator
                            if i not in tensor_mapping:
                                tensor_mapping[i] = {}
                            tensor_mapping[i][weights_index] = new_tensor_index
                        else:
                            # Directly modify the tensor shape since it's only used once
                            weights_tensor.shape = weights_tensor.shape[-2:]

        # Second pass: update operator inputs based on the mapping
        for i, operator in enumerate(subgraph.operators):
            # Check if this is a FullyConnected operator
            opcode = model.operatorCodes[operator.opcodeIndex]
            if opcode.builtinCode == circle.BuiltinOperator.FULLY_CONNECTED:
                # Update inputs according to the mapping
                if i in tensor_mapping:
                    for j, input_idx in enumerate(operator.inputs):
                        if input_idx in tensor_mapping[i]:
                            operator.inputs[j] = tensor_mapping[i][input_idx]
                else:
                    # For tensors that were directly modified, just check if they need updating
                    if len(operator.inputs) >= 2:
                        weights_index = operator.inputs[1]
                        weights_tensor = subgraph.tensors[weights_index]
                        if is_effectively_2d(weights_tensor.shape):
                            # Update the shape to be truly 2D
                            weights_tensor.shape = weights_tensor.shape[-2:]

    # Save the model using utility function
    o2o.save_model_to_stdout(model)


if __name__ == "__main__":
    # Directly invoke processing; I/O handled via stdin/stdout
    modify_fully_connected_weights()
