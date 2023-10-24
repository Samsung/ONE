# How to Cross-build Runtime for AARCH64

In **ONE**, we use `AARCH64` on build files such as Makefile, CMakeLists.txt and so on.

## Prepare AARCH64 Ubuntu RootFS

Install required packages

```
$ sudo apt-get install qemu qemu-user-static binfmt-support debootstrap
```

Use `install_rootfs.sh` script to prepare Root File System. You should have `sudo`

```
$ sudo ./tools/cross/install_rootfs.sh aarch64
```
- supports `arm`(default) and `aarch64` architecutre for now
- supports `bionic`, `focal`, and `jammy` release

To see the options,
```
$ ./tools/cross/install_rootfs.sh -h
```

RootFS will be prepared at `tools/cross/rootfs/aarch64` folder.

***\* CAUTION: The OS version of rootfs must match the OS version of execution target device. On the other hand, you need to match the Ubuntu version of the development PC with the Ubuntu version of rootfs to be used for cross-build. Otherwise, unexpected build errors may occur.***

If you are using Ubuntu 20.04 LTS, select `focal`, if you are using Ubuntu 22.04 LTS, select `jammy`. You can check your Ubuntu code name in the following way.

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
$ ROOTFS_DIR=/home/user/rootfs/aarch64-bionic sudo -E ./tools/cross/install_rootfs.sh aarch64
```

### Using proxy

If you need to use proxy server while building the rootfs, use `--setproxy` option.

```
# for example,
$ sudo ./tools/cross/install_rootfs.sh aarch64 --setproxy="1.2.3.4:8080"
# or
$ sudo ./tools/cross/install_rootfs.sh aarch64 --setproxy="proxy.server.com:8888"
```

This will put `apt` proxy settings in `rootfs/etc/apt/apt.conf.d/90proxy` file
for `http`, `https` and `ftp` protocol.

## Cross build for AARCH64

Install cross compilers
```
$ sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

Give `TARGET_ARCH` variable to set the target architecture
```
$ CROSS_BUILD=1 TARGET_ARCH=aarch64 make
$ CROSS_BUILD=1 TARGET_ARCH=aarch64 make install
```
- supports `armv7l` and `aarch64` for now

If you used `ROOTFS_DIR` to prepare in alternative folder,
you should also give this to makefile.
```
$ CROSS_BUILD=1 ROOTFS_DIR=/home/user/rootfs/aarch64-xenial TARGET_ARCH=aarch64 make
$ CROSS_BUILD=1 ROOTFS_DIR=/home/user/rootfs/aarch64-xenial TARGET_ARCH=aarch64 make install
```
You can also omit the `CROSS_BUILD=1` option if you explicitly pass `ROOTFS_DIR`. In that case, if
the `TARGET_ARCH` are differs from the hostarchitecture, the make script automatically applies
`CROSS_BUILD=1`. So, if you set `ROOTFS_DIR` as an environment variable, you can simply perform
normal build and cross build as follows.

```
$ export ROOTFS_DIR=xxx
...
$ make                         # do normal build
$ TARGET_ARCH=aarch64 make     # do cross build
```

### Run test

To run and test the cross-compiled runtime, you need to copy the compiled output to the target device of the architecture in which it is executable. Please refer to the following document for details on the test procedure. In the guide, `armv7l-linux.debug` in path should be replaced by referring to your build result.

- [Testing neural network model inference with a cross-build runtime](./how-to-cross-build-runtime-for-arm.md#run-test)
