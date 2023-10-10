# nnfwapi package

`nnfwapi` is a package to run `nnpackage` with the nnfw python API.

This package includes the nnfw python API module resulting from runtime build.

It is provided separately for each architecture(x86_64, armv7l, aarch64) from `nnfwapi/libnnfw_api_pybind.py` interface.

## Packaging
Execute this command, then the tasks such as copying modules, and packaging.

```
$ python3 setup.py sdist bdist_wheel
```

## Publishing

To publish the package, twine should be installed.

```
$ sudo apt install twine
```
This command is for TestPyPI only and it will be changed to PyPI.  
Additionally, a private token is required for publishing.

```
$ twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```


## Usage

You can install the package as follows:

```
$ pip install -i https://test.pypi.org/simple/ nnfwapi
```

By specifying the version, you can use a specific version of the package. (recommended)

```
$ pip install -i https://test.pypi.org/simple/ nnfwapi==0.1.1
```

This definition has to be set at the top of the script using nnfw python API.

```
from nnfwapi.libnnfw_api_pybind import *
```


