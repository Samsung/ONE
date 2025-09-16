# onert Python Package

`onert` provides python package to run `nnpackage` package format or supporting modelfile format (`circle`, `tflite`) with the onert's Python API.

This package includes the onert Python API module resulting from runtime build.

It is provided separate package for each architecture(x86_64, aarch64), and experimental cross build target architecture(armv7l).

It uses `onert.infer.session` interface defined in `runtime/onert/api/python/package/infer/session.py`.

## Building the runtime

Prior to the build of the bindings you need to compile and install the part of the project being exposed to Python. You can check the instructions in `docs/howto/how-to-build-runtime.md` but for simplicity you can use the following commands. This demonstrates the build for the x86 architecture, for cross-compilation it is very similar though.

```
$ make -f Makefile.template prepare-buildtool BUILD_TYPE=Release
$ make -f Makefile.template prepare-nncc BUILD_TYPE=Release
$ make -f Makefile.template configure BUILD_TYPE=Release
$ make -f Makefile.template build BUILD_TYPE=Release
$ make -f Makefile.template install BUILD_TYPE=Release
```

After the build is done, the required binaries should be installed in the `Product/out` directory. This is where the native part of the Python bindings is as well. 
You can use a custom installation of the binaries as a starting point of the Python API - in this case specify that location in the `PRODUCT_DIR` environment variable.

## Creating the Python wheel

To create the Python wheel (complete Python API package) execute the following command in the `runtime/infra/python` directory:

```
$ python3 setup.py bdist_wheel --plat-name PLATFORM
```

where `PLATFORM` is one of the supported target architectures: aarch64, x86_64, armv7l

The wheel is then created in the `dist` subdirectory.

## Publishing

We are using github action to publish x86_64 and aarch64 packages to PyPI.
