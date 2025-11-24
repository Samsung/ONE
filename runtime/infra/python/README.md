# onert Python Package

`onert` provides python package to run `nnpackage` package format or supporting modelfile format (`circle`, `tflite`) with the onert's Python API.

This package includes the onert Python API module resulting from runtime build.

It is provided separate package for each architecture(x86_64, aarch64), and experimental cross build target architecture(armv7l).

It uses `onert.infer.session` interface defined in `runtime/onert/api/python/package/infer/session.py`.

## Building the runtime

Prior to the build of the bindings you need to compile and install the part of the project being exposed to Python. You can check the instructions in `docs/howto/how-to-build-runtime.md` but for simplicity you can use the following commands. This demonstrates the build for the x86 architecture, for cross-compilation it is very similar though.

```sh
make -f Makefile.template configure BUILD_TYPE=Release
make -f Makefile.template build BUILD_TYPE=Release
make -f Makefile.template install BUILD_TYPE=Release
```

or in a single command
```sh
make -f Makefile.template configure build install BUILD_TYPE=Release
```

After the build is done, the required binaries should be installed in the `Product/out` directory. This is where the native part of the Python bindings is as well. 
You can use a custom installation of the binaries as a starting point of the Python API - in this case specify that location in the `PRODUCT_DIR` environment variable.

## Creating the Python wheel

To create the Python wheel you will need to install the `build` frontend (check the following link https://pypi.org/project/build/).

Change the working directory to `runtime` and the simplest command you need to issue to build the wheel is:

```sh
python3 -m build --wheel
```

Additionally the build can be parametrized with the following environment variables:

`PLATFORM` - used to build a wheel containing cross compiled binaries. By default the `x86_64` platform is assumed.

`GLIBC_VERSION` - the version of GLIBC used to build the runtime. The expected format is X_Y or X.Y where X and Y are the major and minor versions of GLIBC respectively.

`PRODUCT_DIR` - an alternative location containing the build of the runtime.

An example command that uses the env variables to parametrize the wheel's build:

```sh
PLATFORM=aarch64 GLIBC_VERSION=3_15 python3 -m build --wheel
```

As the result you should be able to find a wheel located in the `dist` directory and the name of this file should be similar to: `onert-0.2.0-cp312-cp312-manylinux_3_15_aarch64.whl`

### Deprecated (but still supported) method
This method of building the wheel is currently deprecated and the approach described above should be used.

To create the Python wheel (complete Python API package) execute the following command in the `runtime/infra/python` directory:

```sh
python3 setup.py bdist_wheel --plat-name manylinux_[X]_[Y]_[PLATFORM]
```

where:
- `[X]` and `[Y]` are the major and minor version of glibc used by the bindings (`ldd --version`)
- `[PLATFORM]` is one of the supported target architectures: aarch64, x86_64, armv7l

An example call should look like this:
```sh
python3 setup.py bdist_wheel --plat-name manylinux_2_35_x86_64
```

The wheel is then created in the `dist` subdirectory.

## Publishing

We are using github action to publish x86_64 and aarch64 packages to PyPI.
