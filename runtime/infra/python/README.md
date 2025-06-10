# onert package

`onert` is a package to run `nnpackage` with the nnfw onert's python API.

This package includes the nnfw-onert python API module resulting from runtime build.

It is provided separate package for each architecture(x86_64, armv7l, aarch64).

It uses `onert/infer.py` interface.

## Packaging
Execute this command, then the tasks such as copying modules, and packaging.

```
$ python3 setup.py sdist bdist_wheel --plat-name manylinux_x_y_arch
```

x_y is the glibc version of manylinux.

arch is supported target architecture: aarch64, x86_64, armv7l

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
$ pip install -i https://test.pypi.org/simple/ nnfw-onert
```

By specifying the version, you can use a specific version of the package. (recommended)

```
$ pip install -i https://test.pypi.org/simple/ nnfw-onert==0.1.1
```

This definition has to be set at the top of the script using nnfw python API.

```
import onert
```

Or you can import the onert module directly.

```
from onert.infer import *
```

This can be use onert session directly.
