from onert import infer
import numpy as np
import sys


def main(nnpackage_path, backends="cpu"):
    # Create session and load nnpackage
    # The default value of backends is "cpu".
    session = infer.session(nnpackage_path, backends)

    # Prepare input. Here we just allocate dummy input arrays.
    input_infos = session.get_inputs_tensorinfo()
    dummy_inputs = []
    for info in input_infos:
        # Create a dummy numpy array filled with zeros.
        dummy_inputs.append(np.zeros(info.shape, dtype=info.dtype))

    outputs = session.infer(dummy_inputs)

    print(f"nnpackage {nnpackage_path.split('/')[-1]} runs successfully.")
    return


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(*argv)
