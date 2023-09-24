# minimal-python

`minimal-python` is a simple driver to run `nnpackage` with nnfw python API.

It takes `nnpackage` as input. It uses **nnfw python API** internally.

It assumes model of float32 tensor type as an input.

## Usage

```
$ python3 minimal.py path_to_nnpackage_directory
```

## Module Description

### Classes

```python

class tensorinfo():
    """
    Tensorinfo describes the type and shape of tensors

    Args:
        dtype (str): The data type
        rank (int): The number of dimensions (rank)
        dims (list): The dimension of tensor. Maximum rank is 6 (NNFW_MAX_RANK).
    """

    def __init__(self, dtype, rank, dims):
        """The constructor of tensorinfo"""
        self.dtype = dtype
        self.rank = rank
        self.dims = dims

    def get_dims(self):
        """Get dims of tensorinfo"""

    def set_dims(self, dims_list):
        """Set dims_list to dims of tensorinfo"""

class nnfw_session():
    """
    Session to query with runtime

    Args:
        nnpackage_path (str): Path to the nnpackage file or unzipped directory to be loaded
        backends (str): Available backends on which nnfw uses
                         Multiple backends can be set and they must be separated by a semicolon
                         (ex: "acl_cl;cpu")
        *op (str): operation to be set
                   Among the multiple backends, the 1st element is used as the default backend.
    """

    def __init__(self, nnpackage_path, backends, *op):
        """
        Create a new session instance, load model from nnpackage file or directory,
        set available backends or the operation's backend and prepare session to be ready for inference
        """
        self.nnpackage_path = nnpackage_path
        self.backends = backends
        self.op = op

    def set_input_tensorinfo(self, index, tensor_info):
        """
        Set input model's tensor info for resizing.

        Args:
            index (int): Index of input to be set (0-indexed)
            tensor_info (tensorinfo): Tensor info to be set
        """

    def run(self):
        """Run inference"""

    def run_async(self):
        """Run inference asynchronously"""

    def await(self):
        """Wait for asynchronous run to finish"""

    def set_input(self, index, buffer):
        """
        Set input buffer

        Args:
            index (int): Index of input to be set (0-indexed)
            buffer (numpy): Raw buffer for input
        """

    def set_output(self, index, buffer):
        """
        Set output buffer

        Args:
            index (int): Index of output to be set (0-indexed)
            buffer (numpy): Raw buffer for output
        """

    def input_size(self):
        """
        Get the number of inputs defined in loaded model

        Returns:
            int: The number of inputs
        """

    def output_size(self):
        """
        Get the number of outputs defined in loaded model

        Returns:
            int: The number of outputs
        """

    def set_input_layout(self, index, *layout):
        """
        Set the layout of an input

        Args:
            index (int): Index of input to be set (0-indexed)
            layout (str): Layout to set to target input
        """

    def set_output_layout(self, index, *layout):
        """
        Set the layout of an output

        Args:
            index (int): Index of output to be set (0-indexed)
            layout (str): Layout to set to target output
        """

    def input_tensorinfo(self, index):
        """
        Get i-th input tensor info

        Args:
            index (int): Index of input

        Returns:
            tensorinfo: Tensor info (shape, type, etc)
        """

    def output_tensorinfo(self, index):
        """
        Get i-th output tensor info

        Args:
            index (int): Index of output

        Returns:
            tensorinfo: Tensor info (shape, type, etc)
        """
```
