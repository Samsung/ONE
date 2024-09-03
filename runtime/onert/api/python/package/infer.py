import numpy as np
import os
import shutil

from .native import libnnfw_api_pybind


def num_elems(tensor_info):
    """Get the total number of elements in nnfw_tensorinfo.dims."""
    n = 1
    for x in range(tensor_info.rank):
        n *= tensor_info.dims[x]
    return n


class session(libnnfw_api_pybind.nnfw_session):
    """Class inherited nnfw_session for easily processing input/output"""
    def __init__(self, nnpackage_path, backends="cpu"):
        super().__init__(nnpackage_path, backends)
        self.inputs = []
        self.outputs = []
        self.set_outputs(self.output_size())

    def set_inputs(self, size, inputs_array=[]):
        """Set inputs for each index"""
        for i in range(size):
            input_tensorinfo = self.input_tensorinfo(i)

            if len(inputs_array) > i:
                input_array = np.array(inputs_array[i], dtype=input_tensorinfo.dtype)
            else:
                print(
                    f"model's input size is {size} but given inputs_array size is {len(inputs_array)}.\n{i}-th index input is replaced by an array filled with 0."
                )
                input_array = np.zeros((num_elems(input_tensorinfo)),
                                       dtype=input_tensorinfo.dtype)

            self.set_input(i, input_array)
            self.inputs.append(input_array)

    def set_outputs(self, size):
        """Set outputs for each index"""
        for i in range(size):
            output_tensorinfo = self.output_tensorinfo(i)
            output_array = np.zeros((num_elems(output_tensorinfo)),
                                    dtype=output_tensorinfo.dtype)
            self.set_output(i, output_array)
            self.outputs.append(output_array)

    def inference(self):
        """Inference model and get outputs"""
        self.run()

        return self.outputs


def tensorinfo():
    return libnnfw_api_pybind.nnfw_tensorinfo()
