# How to Cross-build Runtime for ARM

## Prepare ARM Ubuntu RootFS

Install required packages

```
$ sudo apt-get install qemu-user-static binfmt-support debootstrap
```

Use `install_rootfs.sh` script to prepare Root File System. You should have `sudo`

```
$ sudo ./tools/cross/install_rootfs.sh arm
```
- supports `arm`(default) and `aarch64` architecutre for now
- supports `jammy`, and `noble` release

To see the options,
```
$ ./tools/cross/install_rootfs.sh -h
```

RootFS will be prepared at `tools/cross/rootfs/arm` or `tools/cross/rootfs/aarch64` folder.

***\* CAUTION: The OS version of rootfs must match the OS version of execution target device. On the other hand, you need to match the Ubuntu version of the development PC with the Ubuntu version of rootfs to be used for cross-build. Otherwise, unexpected build errors may occur.***

If you are using Ubuntu 22.04 LTS, select `jammy`, if you are using Ubuntu 24.04 LTS, select `noble`. You can check your Ubuntu code name in the following way.

```
$ cat /etc/lsb-release
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=22.04
DISTRIB_CODENAME=jammy
DISTRIB_DESCRIPTION="Ubuntu 22.04.3 LTS"
```

`install_rootfs.sh` will select development system code name as default.

If a build error occurs because the version of the development system and the target system do not match, and if you can't replace your development system for any reason, you can consider [cross-build using the docker image](how-to-build-runtime-using-prebuilt-docker-image.md).

### Prepare RootFS at alternative folder

Use `ROOTFS_DIR` to a full path to prepare at alternative path.

```
$ ROOTFS_DIR=/home/user/rootfs/arm-jammy sudo -E ./tools/cross/install_rootfs.sh arm
```

### Using proxy

If you need to use proxy server while building the rootfs, use `--setproxy` option.

```
# for example,
$ sudo ./tools/cross/install_rootfs.sh arm --setproxy="1.2.3.4:8080"
# or
$ sudo ./tools/cross/install_rootfs.sh arm --setproxy="proxy.server.com:8888"
```

This will put `apt` proxy settings in `rootfs/etc/apt/apt.conf.d/90proxy` file
for `http`, `https` and `ftp` protocol.

## Install ARM Cross Toolchain

We recommend you have g++ >= 9 installed on your system for c++17.

## Build and install ARM Compute Library

Mostly you only need once of ACL (ARM Compute Library) build.

To build ACL, you need to install scons

```
$ sudo apt-get install scons
```

ACL source will be automatically installed in `externals/ARMCOMPUTE` when you build runtime without any changes.

You can check ACL source information in `infra/cmake/packages/ARMComputeSourceConfig.cmake`

## Install numpy

Python package `numpy` is needed for build. Please use [this guide](how-to-build-runtime.md##install-python-packages) to install it.

## Cross build for ARM by using Makefile.template

Give `TARGET_ARCH` variable to set the target architecture.

If you used `ROOTFS_DIR` to prepare in alternative folder, you should also give this to makefile.

Remember to activate venv if you didn't activate it already:

```
$ source .venv/bin/activate
```

```
(.venv)$ CROSS_BUILD=1 TARGET_ARCH=armv7l make -f Makefile.template

# If ROOTFS_DIR is in alternative folder
(.venv)$ ROOTFS_DIR=/path/to/your/rootfs/arm \
CROSS_BUILD=1 TARGET_ARCH=armv7l make -f Makefile.template
```

Makefile.template will pass crossbuild toolchain setting to cmake automatically by parsing variables.

### Run test

To run and test the cross-compiled runtime, you need to install library packages and copy the compiled output to the target device of the architecture in which it is executable.

1. Install hdf5 library package

```
$ sudo apt install libhdf5-dev
```

2. Copy all artifacts under the `./Product/armv7l-linux.<BUILD_TYPE>` folder to the target device, Odroid-XU4 for example, as a whole.

```
$ ssh odroid mkdir -p one/Product
sjlee@odroid's password:
$ scp -rp ./Product/armv7l-linux.debug odroid:one/Product
sjlee@odroid's password:
FillFrom_runner                                                                                 100%  224KB 223.6KB/s   00:00
benchmark_nnapi.sh                                                                              100% 7464     7.3KB/s   00:00
common.sh                                                                                       100% 2084     2.0KB/s   00:00
test_framework.sh                                                                               100% 3154     3.1KB/s   00:00
test-driver.sh
...
```

3. Log in to the target device, go to the copied path, and reestore the symbolic link settings of the `Product` directory.

```
$ ssh odroid
sjlee@odroid's password:
...
$ cd ~/one/Product
$ ln ${PWD}/armv7l-linux.debug/out out
$ cd ..
$ ls -la Product
drwxrwxr-x  5 sjlee sjlee 4096 Jun  4 20:55 armv7l-linux.debug
lrwxrwxrwx  1 sjlee sjlee   51 Jun  4 20:55 out -> /home/sjlee/one/Product/armv7l-linux.debug/out
```

Now you can test the compilation result in the same way as the native build. Please refer to the following document for details on the test procedure.

- [Testing neural network model inference using the runtime](./how-to-build-runtime.md#run-test)
