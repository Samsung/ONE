# Define the public API of the onert package
# __all__ = ["infer", "train"]
__all__ = ["infer", "tensorinfo", "train"]
# __all__ = ["tensorinfo", "train"]

# Import and expose the infer module's functionalities
from . import infer as infer
# from . import session as infer, tensorinfo

# Import and expose tensorinfo
from .infer import tensorinfo as tensorinfo

# Import and expose the train module's functionalities
# from . import train
from . import train
