# How to Build Compiler

This document is based on the system where Ubuntu Desktop Linux 20.04 LTS is installed with default
settings, and can be applied in other environments without much difference.
For example, Ubuntu Desktop Linux 18.04 LTS environment is also supported but some additional steps may be required.

## Build Requires

If you are going to build this project, the following modules must be installed on your system:

- CMake
- GNU C/C++ compiler (gcc, g++) - the recommended version is GCC 11 or higher. Please check
the [known issues](#known-issues) section for the explanation.

In the Ubuntu, you can easily install it with the following command.

```
$ sudo apt-get install cmake gcc g++
```

If your linux system does not have the basic development configuration, you will need to install
more packages. A list of all packages needed to configure the development environment can be found
in the https://github.com/Samsung/ONE/blob/master/infra/docker/focal/Dockerfile file.

Here is a summary of it

```
$ sudo apt-get install \
build-essential \
cmake \
git \
libboost-all-dev \
libgflags-dev \
libgoogle-glog-dev \
libatlas-base-dev \
libhdf5-dev \
libprotobuf-dev \
protobuf-compiler \
wget \
zip \
unzip \
python3 \
python3-pip \
python3-venv \
python3-dev \
hdf5-tools \
curl
$ pip install numpy flatbuffers
```

Supported platforms:
- Ubuntu 20.04
- Ubuntu 22.04
- Ubuntu 24.04 (experimental)

Supported Python versions:
- Python 3.10 for Ubuntu 20.04 and 22.04
- Python 3.12 for Ubuntu 24.04

Note: Python 3.10 needs to be installed manually on Ubuntu 20.04.
```
sudo apt-get install software-properties-common
sudo -E add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get -y install python3.10 python3.10-venv python3.10-dev
python3.10 -m ensurepip
```

Additional install libtsan_preinit.o manually if you are using Ubuntu 20.04 and gcc-9 (refer https://github.com/Samsung/ONE/issues/11202)
```
$ apt-get download libgcc-10-dev
$ dpkg -x libgcc-10-dev_*_amd64.deb libgcc/
$ sudo cp -f libgcc/usr/lib/gcc/x86_64-linux-gnu/10/libtsan_preinit.o /usr/lib/gcc/x86_64-linux-gnu/9/
$ rm -rf libgcc-10-dev_*_amd64.deb libgcc/
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
$ NNCC_WORKSPACE=build/debug ./nncc build
```
will build debug version in `build/debug` folder, and

```
$ NNCC_WORKSPACE=build/release ./nncc configure -DCMAKE_BUILD_TYPE=Release
$ NNCC_WORKSPACE=build/release ./nncc build
```
will build release version in `build/release` folder.

## Build for Windows

To build for Windows, we use MinGW(Minimalist GNU for Windows). [Here](https://github.com/git-for-windows/build-extra/releases) you can download a tool that includes it.

```
$ git clone https://github.com/Samsung/ONE.git one
$ cd one
$ NNAS_BUILD_PREFIX=build ./nnas create-package --preset 20200731_windows --prefix install
```

- `NNAS_BUILD_PREFIX` is the path to directory where compiler-build-artifacts will be stored.
- `--preset` is the one that specifies a version you will install. You can see `infra/packaging/preset/` directory for more details and getting latest version.
- `--prefix` is the install directory.

## Cross build for Ubuntu/ARM32 (experimental)

Some modules are availble to run in Ubuntu/ARM32 through cross building.

While configuring the build, some modules need to execute tools for generating
test materials and they need to execute in the host(x86-64). So some modules
are needed to build the tools for host before cross building.

Cross build overall steps are like, (1) configure for host
(2) build tools for host (3) configure for ARM32 target (4) and then build
for ARM32 target.

Unit tests can also run in target device.
But value test needs to run TensorFlow lite to get expected results,
and it would be a task to do this so the data files from host execution
are used instead.

Thus to run the unit tests in the target, running in host is needed in prior.

### Prepare root file system

You should prepare Ubuntu/ARM32 root file system for cross compilation.
Please refer
[how-to-cross-build-runtime-for-arm.md](how-to-cross-build-runtime-for-arm.md)
for preparation.

You can set `ROOTFS_ARM` environment variable if you have in alternative
folder.

### Clean existing external source for patches

Some external projects from source are not "cross compile ready with CMake"
projects. This experimental project prepared some patches for this.
Just remove the source and stamp file like below and the `make` will prepare
patch applied source codes.
```
rm -rf externals/HDF5
rm -rf externals/PROTOBUF
rm externals/HDF5.stamp
rm externals/PROTOBUF.stamp
```

### Build

To cross build, `infra/nncc/Makefile.arm32` file is provided as an example to
work with `make` command.
```
make -f infra/nncc/Makefile.arm32 cfg
make -f infra/nncc/Makefile.arm32 debug
```
First `make` will run above steps (1), (2) and (3). Second `make` will run (4).

### Test

Preprequisite for testing in ARM32 device.
```
# numpy is required for value match in ARM32 target device
sudo apt-get install python3-pip
python3 -m pip install numpy
```

You can also run unit tests in ARM32 Ubuntu device with cross build results.
First you need to run the test in host to prepare files that are currently
complicated in target device.
For value test with python, separate venv is requried. make target `test_venv`
will prepare this.
```
# run this in x86-64 host
make -f infra/nncc/Makefile.arm32 test_prep

# run this in ARM32 target device
make -f infra/nncc/Makefile.arm32 test_venv
make -f infra/nncc/Makefile.arm32 test
```

NOTE: this assumes
- host and target have same directoy structure
- should copy `build` folder to target or
- mounting `ONE` folder with NFS on the target would be simple

## Known issues
There's a potential known build error when attempting to cross-compile for ARM32 using GCC 10.5.
You might encounter an error:
`comparison of unsigned expression in ‘< 0’ is always false [-Werror=type-limits]` reported from
the `CircleNodeMixins.h` file. This is likely GCC's bug in this specific version.
There's a workaround for it though - you can apply the following changes to both for loops
in the `CircleNodeMixins.h`:

```
FixedArityNode()
{
  if constexpr (N > 0)
  {
    _args.resize(N);
    for (uint32_t n = 0; n < N; ++n)
    {
    _args[n] = std::make_unique<loco::Use>(this);
    }
  }
}

void drop(void) final
{
  if constexpr (N > 0)
  {
    for (uint32_t n = 0; n < N; ++n)
    {
      _args.at(n)->node(nullptr);
    }
  }
}
```
