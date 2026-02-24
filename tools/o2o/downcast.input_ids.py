#!/usr/bin/env python3

import o2o
import circle

# Import specific Circle types for better type annotations
from circle import (TensorT, OperatorT, SubGraphT, ModelT, BufferT, OperatorCodeT,
                    BuiltinOperator, TensorType)


def retype_input_ids():
    """Main function to change input_ids tensor type from int64 to int32"""
    # Load the model using utility function
    model = o2o.load_model_from_stdin()

    def process_subgraph(subgraph):
        """Process a single subgraph to find and retype input_ids tensors"""
        o2o.log(f"Processing subgraph with {len(subgraph.tensors)} tensors")

        retyped_count = 0

        # Collect subgraph inputs for quick lookup
        subgraph_inputs = set(subgraph.inputs)

        for op_idx, op in enumerate(subgraph.operators):
            opcode = model.operatorCodes[op.opcodeIndex]

            if opcode.builtinCode == circle.BuiltinOperator.GATHER:
                # GATHER input 1 is the indices tensor (params, indices)
                if op.inputs is not None and len(op.inputs) > 1:
                    input_tensor_idx = op.inputs[1]

                    # Check if this input is a subgraph input
                    if input_tensor_idx in subgraph_inputs:
                        tensor = subgraph.tensors[input_tensor_idx]

                        # Check if type is INT64
                        if tensor.type == circle.TensorType.INT64:
                            tensor_name = o2o.get_tensor_name(tensor)
                            old_type = "int64"
                            new_type = "int32"

                            # Change type to int32
                            tensor.type = circle.TensorType.INT32

                            o2o.log(
                                f"Retyped tensor: {tensor_name} (Index: {input_tensor_idx}) {old_type} → {new_type}"
                            )
                            retyped_count += 1

        if retyped_count > 0:
            o2o.log(f"Retyped {retyped_count} input_ids tensors in this subgraph")
        else:
            o2o.log("No input_ids tensors were retyped in this subgraph")

        return retyped_count > 0, retyped_count

    # Process all subgraphs using utility function
    overall_modified, total_changes = o2o.process_subgraphs(model, process_subgraph)

    if not overall_modified:
        o2o.log("No input_ids tensors were modified.")

    # Save the model using utility function
    o2o.save_model_to_stdout(model)


if __name__ == "__main__":
    # Directly invoke processing; I/O handled via stdin/stdout
    retype_input_ids()
