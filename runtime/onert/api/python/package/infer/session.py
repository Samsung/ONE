from ..native import libnnfw_api_pybind
from ..common.basesession import BaseSession


class session(BaseSession):
    """
    Class for inference using nnfw_session.
    """
    def __init__(self, path: str = None, backends: str = "cpu"):
        """
        Initialize the inference session.
        Args:
            path (str): Path to the model file or nnpackage directory.
            backends (str): Backends to use, default is "cpu".
        """
        if path is not None:
            super().__init__(libnnfw_api_pybind.infer.nnfw_session(path, backends))
            self.session.prepare()
            self.set_outputs(self.session.output_size())
        else:
            super().__init__()

    def compile(self, path: str, backends: str = "cpu"):
        """
        Prepare the session by recreating it with new parameters.
        Args:
            path (str): Path to the model file or nnpackage directory. Defaults to the existing path.
            backends (str): Backends to use. Defaults to the existing backends.
        """
        # Update parameters if provided
        if path is None:
            raise ValueError("path must not be None.")
        # Recreate the session with updated parameters
        self._recreate_session(libnnfw_api_pybind.infer.nnfw_session(path, backends))
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
