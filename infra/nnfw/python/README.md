# nnfwapi package

`nnfwapi` is a package to run `nnpackage` with nnfw python API.

This package includes nnfw python API module resulted from runtime build.

It is provided separately according to each architecture(x86_64, armv7l, aarch64) from `nnfwapi/libnnfw_api_pybind.py` interface.

## Requirement

If you don't have any libhdf5-dev.so error, it's okay not to install `libhdf5-dev`.

```
$ sudo apt install libhdf5-dev python3 python3-pip
```

`nnfwapi` should be installed from PyPI.

```
$ pip install nnfwapi
```

By specifying the version, you can use a specific version of the package.

```
$ pip install nnfwapi==0.1.0
```

## Usage

This definition have to be set on the top of the script used nnfw python API.

```
from nnfwapi.libnnfw_api_pybind import *
```

## Packaging and Publishing

`pack.sh` is written for publishing this package to PyPI easily.

Execute this command, then the tasks such as copying modules, packaging, and publishing are automatically performed.

```
sh pack.sh
```

Before publishing, be carefule to edit the information on `setup.py` such as version, description, etc.
