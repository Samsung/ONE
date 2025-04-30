from onert import infer
import numpy as np
import sys


def main(nnpackage_path, backends="cpu"):
    # Create session and load the nnpackage
    sess = infer.session(nnpackage_path, backends)

    # Retrieve the current tensorinfo for all inputs.
    current_input_infos = sess.get_inputs_tensorinfo()

    # Create new tensorinfo objects with a static shape modification.
    # For this example, assume we change the first dimension (e.g., batch size) to 10.
    new_input_infos = []
    for info in current_input_infos:
        # For example, if the current shape is (?, 4), update it to (10, 4).
        # We copy the current info and modify the rank and dims.
        # (Note: Depending on your model, you may want to modify additional dimensions.)
        new_shape = [10] + list(info.dims[1:info.rank])
        info.dims = new_shape
        new_input_infos.append(info)

    # Update all input tensorinfos in the session at once.
    # This will call prepare() and set_outputs() internally.
    sess.update_inputs_tensorinfo(new_input_infos)

    # Create dummy input arrays based on the new static shapes.
    dummy_inputs = []
    for info in new_input_infos:
        # Build the shape tuple from tensorinfo dimensions.
        shape = tuple(info.dims[:info.rank])
        # Create a dummy numpy array filled with zeros.
        dummy_inputs.append(np.zeros(shape, dtype=info.dtype))

    # Run inference with the new static input shapes.
    outputs = sess.infer(dummy_inputs)

    print(
        f"Static shape modification sample: nnpackage {nnpackage_path.split('/')[-1]} runs successfully."
    )
    return


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(*argv)
