import numpy as np

from ..native import libnnfw_api_pybind
from ..common.basesession import BaseSession


class session(BaseSession):
    """
    Class for inference using nnfw_session.
    """
    def __init__(self, nnpackage_path: str = None, backends: str = "cpu"):
        """
        Initialize the inference session.

        Args:
            nnpackage_path (str): Path to the nnpackage file or directory.
            backends (str): Backends to use, default is "cpu".
        """
        if nnpackage_path is not None:
            super().__init__(
                libnnfw_api_pybind.infer.nnfw_session(nnpackage_path, backends))
            self.session.prepare()
            self.set_outputs(self.session.output_size())
        else:
            super().__init__()

    def compile(self, nnpackage_path: str, backends: str = "cpu"):
        """
        Prepare the session by recreating it with new parameters.

        Args:
            nnpackage_path (str): Path to the nnpackage file or directory. Defaults to the existing path.
            backends (str): Backends to use. Defaults to the existing backends.
        """
        # Update parameters if provided
        if nnpackage_path is None:
            raise ValueError("nnpackage_path must not be None.")

        # Recreate the session with updated parameters
        self._recreate_session(
            libnnfw_api_pybind.infer.nnfw_session(nnpackage_path, backends))

        # Prepare the new session
        self.session.prepare()
        self.set_outputs(self.session.output_size())

    def inference(self):
        """
        Perform model and get outputs

        Returns:
            list: Outputs from the model.
        """
        self.session.run()
        return self.outputs


def tensorinfo():
    return libnnfw_api_pybind.infer.nnfw_tensorinfo()
