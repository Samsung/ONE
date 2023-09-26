from nnfwapi.libnnfw_api_pybind import *
import numpy as np
import sys


def num_elems(tensor_info):
    """Get the total number of elements in nnfw_tensorinfo.dims."""
    n = 1
    for x in range(tensor_info.rank):
        n *= tensor_info.dims[x]
    return n


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
    inputs = []

    for i in range(input_size):
        # Get i-th input's info
        input_tensorinfo = session.input_tensorinfo(i)
        ti_dtype = input_tensorinfo.dtype

        input_array = [0.] * num_elems(input_tensorinfo)
        input_array = np.array(input_array, dtype=ti_dtype)
        # TODO: Please add initialization for your input.
        session.set_input(i, input_array)

        inputs.append(input_array)

    # Prepare output
    output_size = session.output_size()
    outputs = []

    for i in range(output_size):
        # Get i-th output's info
        output_tensorinfo = session.output_tensorinfo(i)
        ti_dtype = output_tensorinfo.dtype

        output_array = [0.] * num_elems(output_tensorinfo)
        output_array = np.array(output_array, dtype=ti_dtype)
        session.set_output(i, output_array)

        outputs.append(output_array)

    # Do inference
    session.run()

    # TODO: Please print or compare the output value in your way.

    print(f"nnpackage {nnpackage_path.split('/')[-1]} runs successfully.")
    return


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(*argv)
