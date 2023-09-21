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

def nnfw_session():
    return libnnfw_api_pybind.nnfw_session

def tensorinfo():
    return libnnfw_api_pybind.nnfw_tensorinfo()

