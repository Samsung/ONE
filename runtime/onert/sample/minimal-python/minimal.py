from nnfwapi.libnnfw_api_pybind import *
import numpy as np
import sys


def num_elems(tensor_info):
    n = 1
    for x in range(tensor_info.rank):
        n *= tensor_info.dims[x]
    return n


def main(NNPACKAGE_PATH, BACKEND="cpu", OPERATION=""):
    # OPERATION is optional for assigning a specific backend to operations.
    # "cpu" is default value of BACKEND.
    if OPERATION:
        session = nnfw_session(NNPACKAGE_PATH, BACKEND, OPERATION)
    else:
        session = nnfw_session(NNPACKAGE_PATH, BACKEND)

    input_size = session.input_size()
    inputs = []

    for i in range(input_size):
        input_tensorinfo = session.input_tensorinfo(i)
        ti_dtype = input_tensorinfo.dtype

        input_array = [0.] * num_elems(input_tensorinfo)
        input_array = np.array(input_array, dtype=ti_dtype)
        session.set_input(i, input_array)

        inputs.append(input_array)

    output_size = session.output_size()
    outputs = []

    for i in range(output_size):
        output_tensorinfo = session.output_tensorinfo(i)
        ti_dtype = output_tensorinfo.dtype

        output_array = [0.] * num_elems(output_tensorinfo)
        output_array = np.array(output_array, dtype=ti_dtype)
        session.set_output(i, output_array)

        outputs.append(output_array)

    # Do inference
    session.run()

    session.close_session()

    print(f"nnpackage {NNPACKAGE_PATH.split('/')[-1]} runs successfully.")
    return


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(*argv)
