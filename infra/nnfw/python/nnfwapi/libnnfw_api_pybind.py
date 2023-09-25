import platform
import os
import shutil

#__all__ = ['libnnfw_api_pybind']
# check each architecture
architecture = platform.machine()

if architecture == 'x86_64':
    from .x86_64 import libnnfw_api_pybind
elif architecture == 'armv7l':
    from .armv7l import libnnfw_api_pybind
elif architecture == 'aarch64':
    from .aarch64 import libnnfw_api_pybind
else:
    raise ImportError(f"Unsupported architecture: {architecture}")


def nnfw_session(*args):
    num = len(args)
    if (num == 2):
        return libnnfw_api_pybind.nnfw_session(args[0], args[1])
    elif (num == 3):
        return libnnfw_api_pybind.nnfw_session(args[0], args[2], args[1])
    else:
        print("Syntax Error")
        return


def tensorinfo():
    return libnnfw_api_pybind.nnfw_tensorinfo()
