# How to Build Compiler

This document is based on the system where Ubuntu Desktop Linux 18.04 LTS is installed with default
settings, and can be applied in other environments without much difference. For reference, the
development of our project started in the Ubuntu Desktop Linux 16.04 LTS environment.
As of now, to build in 16.04, please use gcc 7.x or above.

## Build Requires

If you are going to build this project, the following modules must be installed on your system:

- CMake
- Boost C++ libraries

In the Ubuntu, you can easily install it with the following command.

```
$ sudo apt-get install cmake libboost-all-dev
```

If your linux system does not have the basic development configuration, you will need to install
more packages. A list of all packages needed to configure the development environment can be found
in the https://github.com/Samsung/ONE/blob/master/infra/docker/Dockerfile.1804 file.

Here is a summary of it

```
$ sudo apt-get install \
build-essential \
clang-format-3.9 \
cmake \
doxygen \
git \
hdf5-tools \
lcov \
libatlas-base-dev \
libboost-all-dev \
libgflags-dev \
libgoogle-glog-dev \
libgtest-dev \
libhdf5-dev \
libprotobuf-dev \
protobuf-compiler \
pylint \
python3 \
python3-pip \
python3-venv \
scons \
software-properties-common \
unzip \
wget

$ mkdir /tmp/gtest
$ cd /tmp/gtest
$ cmake /usr/src/gtest
$ make
$ sudo mv *.a /usr/lib

$ pip install yapf==0.22.0 numpy
```

## Build for Ubuntu

In a typical linux development environment, including Ubuntu, you can build the compiler with a
simple command like this:

```
$ git clone https://github.com/Samsung/ONE.git one
$ cd one
$ ./nncc configure
$ ./nncc build
```
Build artifacts will be placed in `build` folder.

To run unit tests:
```
$ ./nncc test
```

Above steps will build all the modules in the compiler folder. There are modules that are currently
not active. To build only as of now active modules of the compiler, we provide a preset of modules
to build with below command:
```
$ ./nnas create-package --prefix $HOME/.local
```

With this command, `~/.local` folder will contain all files in release.
If you have added `~/.local/bin` in PATH, then you will now have latest compiler binaries.

### Build for debug and release separately

Build target folder can be customized by `NNCC_WORKSPACE` environment, as we may want to separate
debug and release builds.

```
$ NNCC_WORKSPACE=build/debug ./nncc configure
$ ./nncc build
```
will build debug version in `build/debug` folder, and

```
$ NNCC_WORKSPACE=build/release ./nncc configure -DCMAKE_BUILD_TYPE=Release
$ ./nncc build
```
will build release version in `build/release` folder.

### Trouble shooting

If you are using python3.8, as there is no TensorFlow1.13.2 package for python3.8, build may fail.
Please install python3.7 or lower versions as default python3.

## Build for Windows

To build for Windows, we use MinGW(Minimalist GNU for Windows). [Here](https://github.com/git-for-windows/build-extra/releases) you can download a tool that includes it.

```
$ git clone https://github.com/Samsung/ONE.git one
$ cd one
$ NNAS_BUILD_PREFIX=build ./nnas create-package --preset 20200731_windows --prefix install
```

- `NNAS_BUILD_PREFIX` is the path to directory where compiler-build-artifacts will be stored.
- `--preset` is the one that specifies a version you will be install. You can see `infra/packaging/preset/` directory for more details and getting latest version.
- `--prefix` is the install directory.
