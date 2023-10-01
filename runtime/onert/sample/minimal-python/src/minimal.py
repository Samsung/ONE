from nnfwapi.libnnfw_api_pybind import *
import sys


def main(nnpackage_path, backends="cpu", operations=""):
    # Create session and load nnpackage
    # operations is optional to assign a specific backends to each operation.
    # The default value of backends is "cpu".
    if operations:
        session = nnfw_session(nnpackage_path, backends, operations)
    else:
        session = nnfw_session(nnpackage_path, backends)

    # Prepare input. Here we just allocate dummy input arrays.
    input_size = session.input_size()
    session.set_inputs(input_size)

    outputs = session.inference()

    print(f"nnpackage {nnpackage_path.split('/')[-1]} runs successfully.")
    return


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(*argv)
