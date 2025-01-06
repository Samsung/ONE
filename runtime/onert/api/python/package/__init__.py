# Define the public API of the onert package
__all__ = ["infer", "tensorinfo"]

# Import and expose the infer module's functionalities
from . import infer

# Import and expose tensorinfo
from .common import tensorinfo as tensorinfo
