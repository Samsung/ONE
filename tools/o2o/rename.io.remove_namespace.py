#!/usr/bin/env python3

import sys
import circle
import flatbuffers
from typing import Tuple
import o2o

# Import specific Circle types for better type annotations
from circle import (TensorT, OperatorT, SubGraphT, ModelT, BufferT, OperatorCodeT,
                    BuiltinOperator, TensorType)


def load_model_from_stdin() -> 'circle.ModelT':
    """Load a Circle model from binary data read from stdin."""
    data = sys.stdin.buffer.read()
    buf = bytearray(data)
    model = circle.Model.GetRootAsModel(buf, 0)
    model = circle.ModelT.InitFromObj(model)
    return model


def save_model_to_stdout(model: 'circle.ModelT'):
    """Serialize a Circle model and write it to stdout as binary data."""
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), b'CIR0')
    sys.stdout.buffer.write(builder.Output())


def remove_namespace_from_inputs_and_outputs(model: 'circle.ModelT'):
    """Remove namespace from tensor names within the given model."""
    pattern = r'(.*)::(.*)'

    def process_subgraph(subgraph):
        """Process a single subgraph, renaming matching tensor names."""
        o2o.log(
            f"Processing subgraph with {len(subgraph.inputs)} inputs and {len(subgraph.outputs)} outputs"
        )
        renamed_count = 0

        # Process input tensors
        for input_tensor_index in subgraph.inputs:
            tensor = subgraph.tensors[input_tensor_index]
            was_renamed, old_name, new_name = o2o.rename_tensor_if_matches(
                tensor, pattern, lambda match: match.group(2))
            if was_renamed:
                o2o.log(f"Renaming input tensor: {old_name} → {new_name}")
                renamed_count += 1

        # Process output tensors
        for output_tensor_index in subgraph.outputs:
            tensor = subgraph.tensors[output_tensor_index]
            was_renamed, old_name, new_name = o2o.rename_tensor_if_matches(
                tensor, pattern, lambda match: match.group(2))
            if was_renamed:
                o2o.log(f"Renaming output tensor: {old_name} → {new_name}")
                renamed_count += 1

        if renamed_count > 0:
            o2o.log(f"Renamed {renamed_count} input/output tensors in this subgraph")
        else:
            o2o.log("No input/output tensors were renamed in this subgraph")

        return renamed_count > 0, renamed_count

    overall_modified, total_changes = o2o.process_subgraphs(model, process_subgraph)

    if not overall_modified:
        o2o.log("No tensors were modified.")
    else:
        o2o.log(f"Total tensors renamed across all subgraphs: {total_changes}")


def main():
    """Entry point: read model from stdin, process, write to stdout."""
    model = load_model_from_stdin()
    remove_namespace_from_inputs_and_outputs(model)
    save_model_to_stdout(model)


if __name__ == "__main__":
    main()
