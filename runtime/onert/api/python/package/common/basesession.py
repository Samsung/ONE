import numpy as np

from ..native import libnnfw_api_pybind


def num_elems(tensor_info):
    """Get the total number of elements in nnfw_tensorinfo.dims."""
    n = 1
    for x in range(tensor_info.rank):
        n *= tensor_info.dims[x]
    return n


class BaseSession:
    """
    Base class providing common functionality for inference and training sessions.
    """
    def __init__(self, backend_session):
        """
        Initialize the BaseSession with a backend session.
        Args:
            backend_session: A backend-specific session object (e.g., nnfw_session).
        """
        self.session = backend_session
        self.inputs = []
        self.outputs = []

    def __getattr__(self, name):
        """
        Delegate attribute access to the bound NNFW_SESSION instance.
        Args:
            name (str): The name of the attribute or method to access.
        Returns:
            The attribute or method from the bound NNFW_SESSION instance.
        """
        return getattr(self.session, name)

    def set_inputs(self, size, inputs_array=[]):
        """
        Set the input tensors for the session.
        Args:
            size (int): Number of input tensors.
            inputs_array (list): List of numpy arrays for the input data.
        """
        for i in range(size):
            input_tensorinfo = self.session.input_tensorinfo(i)

            if len(inputs_array) > i:
                input_array = np.array(inputs_array[i], dtype=input_tensorinfo.dtype)
            else:
                print(
                    f"Model's input size is {size}, but given inputs_array size is {len(inputs_array)}.\n{i}-th index input is replaced by an array filled with 0."
                )
                input_array = np.zeros((num_elems(input_tensorinfo)),
                                       dtype=input_tensorinfo.dtype)

            self.session.set_input(i, input_array)
            self.inputs.append(input_array)

    def set_outputs(self, size):
        """
        Set the output tensors for the session.
        Args:
            size (int): Number of output tensors.
        """
        for i in range(size):
            output_tensorinfo = self.session.output_tensorinfo(i)
            output_array = np.zeros((num_elems(output_tensorinfo)),
                                    dtype=output_tensorinfo.dtype)
            self.session.set_output(i, output_array)
            self.outputs.append(output_array)


def tensorinfo():
    return libnnfw_api_pybind.infer.nnfw_tensorinfo()
