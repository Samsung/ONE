#!/usr/bin/env python3

import o2o


def reorder_output_tensors():
    """Reorder output tensors: move tensor 0 to the end, shift others forward"""
    o2o.log("Loading model from stdin")
    model = o2o.load_model_from_stdin()

    if not model.subgraphs:
        o2o.log("Model has no subgraphs. Exiting.")
        o2o.save_model_to_stdout(model)
        return

    for subgraph_idx, subgraph in enumerate(model.subgraphs):
        if len(subgraph.outputs) <= 1:
            o2o.log(
                f"Subgraph {subgraph_idx}: Only {len(subgraph.outputs)} output tensor(s), no reordering needed"
            )
            continue

        # Convert numpy array to Python list for proper concatenation
        original_outputs = subgraph.outputs.copy()
        outputs_list = original_outputs.tolist()

        # Move first output tensor to the end
        # Original: [a, b, c, d] -> New: [b, c, d, a]
        first_output = outputs_list[0]
        other_outputs = outputs_list[1:]
        new_outputs = other_outputs + [first_output]

        subgraph.outputs = new_outputs
        o2o.log(
            f"Subgraph {subgraph_idx}: Reordered outputs {original_outputs.tolist()} -> {new_outputs}"
        )

    o2o.save_model_to_stdout(model)


if __name__ == "__main__":
    # Directly invoke processing; I/O handled via stdin/stdout
    reorder_output_tensors()
