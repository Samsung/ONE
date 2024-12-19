import numpy as np

from ..native import libnnfw_api_pybind
from ..common.basesession import BaseSession


class session(BaseSession):
    """
    Class for inference using nnfw_session.
    """
    def __init__(self, nnpackage_path, backends="cpu"):
        """
        Initialize the inference session.

        Args:
            nnpackage_path (str): Path to the nnpackage file or directory.
            backends (str): Backends to use, default is "cpu".
        """
        super().__init__(libnnfw_api_pybind.infer.nnfw_session(nnpackage_path, backends))
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
