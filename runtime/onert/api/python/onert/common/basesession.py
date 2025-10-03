from typing import List
import numpy as np

from ..native.libnnfw_api_pybind import infer, tensorinfo
from ..native.libnnfw_api_pybind.exception import OnertError


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
    def __init__(self, backend_session=None):
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
        if name in self.__dict__:
            # First, try to get the attribute from the instance's own dictionary
            return self.__dict__[name]
        elif hasattr(self.session, name):
            # If not found, delegate to the session object
            return getattr(self.session, name)
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'")

    def _recreate_session(self, backend_session):
        """
        Protected method to recreate the session.
        Subclasses can override this method to provide custom session recreation logic.
        """
        if self.session is not None:
            del self.session  # Clean up the existing session
        self.session = backend_session

    def get_inputs_tensorinfo(self) -> List[tensorinfo]:
        """
        Retrieve tensorinfo for all input tensors.

        Raises:
            OnertError: If the underlying C-API call fails.

        Returns:
            list[tensorinfo]: A list of tensorinfo objects for each input.
        """
        num_inputs: int = self.session.input_size()
        infos: List[tensorinfo] = []
        for i in range(num_inputs):
            try:
                infos.append(self.session.input_tensorinfo(i))
            except ValueError:
                raise
            except Exception as e:
                raise OnertError(f"Failed to get input tensorinfo #{i}: {e}") from e
        return infos

    def get_outputs_tensorinfo(self) -> List[tensorinfo]:
        """
        Retrieve tensorinfo for all output tensors.

        Raises:
            OnertError: If the underlying C-API call fails.

        Returns:
            list[tensorinfo]: A list of tensorinfo objects for each output.
        """
        num_outputs: int = self.session.output_size()
        infos: List[tensorinfo] = []
        for i in range(num_outputs):
            try:
                infos.append(self.session.output_tensorinfo(i))
            except ValueError:
                raise
            except Exception as e:
                raise OnertError(f"Failed to get output tensorinfo #{i}: {e}") from e
        return infos

    def set_inputs(self, size, inputs_array=[]):
        """
        Set the input tensors for the session.

        Args:
            size (int): Number of input tensors.
            inputs_array (list): List of numpy arrays for the input data.

        Raises:
            ValueError: If session uninitialized.
            OnertError:  If any C-API call fails.
        """
        if self.session is None:
            raise ValueError(
                "Session is not initialized with a model. Please compile with a model before setting inputs."
            )

        self.inputs = []
        for i in range(size):
            try:
                input_tensorinfo = self.session.input_tensorinfo(i)
            except ValueError:
                raise
            except Exception as e:
                raise OnertError(f"Failed to get input tensorinfo #{i}: {e}") from e

            if len(inputs_array) > i:
                input_array = np.array(inputs_array[i], dtype=input_tensorinfo.dtype)
            else:
                print(
                    f"Model's input size is {size}, but given inputs_array size is {len(inputs_array)}.\n{i}-th index input is replaced by an array filled with 0."
                )
                input_array = np.zeros((num_elems(input_tensorinfo)),
                                       dtype=input_tensorinfo.dtype)

            # Check if the shape of input_array matches the dims of input_tensorinfo
            if input_array.shape != tuple(input_tensorinfo.dims):
                # If not, set the input tensor info to match the input_array shape
                try:
                    input_tensorinfo.rank = len(input_array)
                    input_tensorinfo.dims = list(input_array.shape)
                    self.session.set_input_tensorinfo(i, input_tensorinfo)
                except Exception as e:
                    raise OnertError(f"Failed to set input tensor info #{i}: {e}") from e

            try:
                self.session.set_input(i, input_array)
            except ValueError:
                raise
            except Exception as e:
                raise OnertError(f"Failed to set input #{i}: {e}") from e

            self.inputs.append(input_array)

    def _set_outputs(self, size):
        """
        Set the output tensors for the session.

        Args:
            size (int): Number of output tensors.

	    Raises:
            ValueError: If session uninitialized.
            OnertError:  If any C-API call fails.
        """
        if self.session is None:
            raise ValueError(
                "Session is not initialized with a model. Please compile a model before setting outputs."
            )

        self.outputs = []
        for i in range(size):
            try:
                output_array = self.session.get_output(i)
            except ValueError:
                raise
            except Exception as e:
                raise OnertError(f"Failed to get output #{i}: {e}") from e

            self.outputs.append(output_array)


def tensorinfo():
    """
    Shortcut to create a fresh tensorinfo instance.
    Raises:
        OnertError: If the C-API call fails.
    """

    try:
        return infer.nnfw_tensorinfo()
    except OnertError:
        raise
    except Exception as e:
        raise OnertError(f"Failed to create tensorinfo: {e}") from e
