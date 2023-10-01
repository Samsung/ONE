# minimal-python

`minimal-python` is a simple driver to run `nnpackage` with nnfw python API.

It takes `nnpackage` as input. It uses **nnfw python API** internally.

It assumes model of float32 tensor type as an input.

## Usage

```
$ python3 minimal.py path_to_nnpackage_directory
```

## Classes

```python
class nnfw_session():
    """
    Session to query with runtime
    Attributes:
        nnpackage_path (str): Path to the nnpackage file or unzipped directory to be loaded
        backends (str): Available backends on which nnfw uses (cpu, acl_cl, acl_neon)
                        Multiple backends can be set and they must be separated by a semicolon (ex: "acl_cl;cpu")
                        Among the multiple backends, the 1st element is used as the default backend.
        op (str): operation to be set
        inputs (list): list containing input for each index
        outputs (list): lsit containing output for each index
    """

    def __init__(self, nnpackage_path, backends="cpu", operations=""):
        """
        Create a new session instance, load model from nnpackage file or directory,
        set available backends or the operation's backend and prepare session to be ready for inference
        Set output for each index
        """
        if operations:
            super().__init__(nnpackage_path, operations, backends)
        else:
            super().__init__(nnpackage_path, backends)

        self.inputs = []
        self.outputs = []
        self.set_outputs(self.output_size())

    def set_inputs(self, size, inputs_array=[]):
        """
        Set input buffer for each index
        Args:
            size (int): Input size of model
            inputs_array (list): List containing input to set for each index
        """

    def set_outputs(self, size):
        """
        Set output buffer for each index
        Args:
            size (int): Output size of model
        """

    def inference(self):
        """Run inference and get outputs

        Returns:
            list: Outputs from running inference
        """
```
