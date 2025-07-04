# onert Python Package

`onert` provides python package to run `nnpackage` package format or supporting modelfile format (`circle`, `tflite`) with the onert's python API.

This package includes the onert python API module resulting from runtime build.

It is provided separate package for each architecture(x86_64, aarch64), and experimental cross build target architecture(armv7l).

It uses `onert.infer.session` interface defined in `runtime/onert/api/python/package/infer/session.py`.

## Packaging

Execute this command, then the tasks such as copying modules, and packaging.

```
$ python3 setup.py sdist bdist_wheel --plat-name manylinux_x_y_arch
```

x_y is the glibc version of manylinux.

arch is supported target architecture: aarch64, x86_64, armv7l

## Publishing

We are using github action to publish x86_64 and aarch64 packages to PyPI.
