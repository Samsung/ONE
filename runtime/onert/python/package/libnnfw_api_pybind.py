import numpy as np
import platform
import os
import shutil

#__all__ = ['libnnfw_api_pybind']
# check each architecture
architecture = platform.machine()

if architecture == 'x86_64':
    from .x86_64 import libnnfw_api_pybind
elif architecture == 'armv7l':
    from .armv7l import libnnfw_api_pybind
elif architecture == 'aarch64':
    from .aarch64 import libnnfw_api_pybind
else:
    raise ImportError(f"Unsupported architecture: {architecture}")


def num_elems(tensor_info):
    """Get the total number of elements in nnfw_tensorinfo.dims."""
    n = 1
    for x in range(tensor_info.rank):
        n *= tensor_info.dims[x]
    return n


class nnfw_session_wrapper(libnnfw_api_pybind.nnfw_session):
    """Class inherited nnfw_session for easily processing input/output"""

    def __init__(self, nnpackage_path, backends="cpu", operations=""):
        if operations:
            super().__init__(nnpackage_path, operations, backends)
        else:
            super().__init__(nnpackage_path, backends)

        self.inputs = []
        self.outputs = []
        self.set_outputs(self.output_size())

    def set_inputs(self, size, inputs_array=[]):
        """Set inputs for each index"""
        for i in range(size):
            input_tensorinfo = self.input_tensorinfo(i)
            ti_dtype = input_tensorinfo.dtype

            if len(inputs_array) > i:
                input_array = inputs_array[i]
            else:
                print(
                    f"model's input size is {size} but given inputs_array size is {len(inputs_array)}.\n{i}-th index input is replaced by an array filled with 0."
                )
                input_array = [0.] * num_elems(input_tensorinfo)

            input_array = np.array(input_array, dtype=ti_dtype)
            self.set_input(i, input_array)

            self.inputs.append(input_array)

    def set_outputs(self, size):
        """Set outputs for each index"""
        for i in range(size):
            output_tensorinfo = self.output_tensorinfo(i)
            ti_dtype = output_tensorinfo.dtype

            output_array = [0.] * num_elems(output_tensorinfo)
            output_array = np.array(output_array, dtype=ti_dtype)
            self.set_output(i, output_array)

            self.outputs.append(output_array)

    def inference(self):
        """Inference model and get outputs"""
        self.run()

        return self.outputs


def nnfw_session(nnpackage_path, backends="cpu", operations=""):
    if operations == "":
        return nnfw_session_wrapper(nnpackage_path, backends)
    elif operations:
        return nnfw_session_wrapper(nnpackage_path, backends, operations)
    else:
        print("Syntax Error")
        return


def tensorinfo():
    return libnnfw_api_pybind.nnfw_tensorinfo()
