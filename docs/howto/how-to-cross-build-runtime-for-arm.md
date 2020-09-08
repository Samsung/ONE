# How to Cross-build Runtime for ARM

## Prepare ARM Ubuntu RootFS

Install required packages

```
$ sudo apt-get install qemu qemu-user-static binfmt-support debootstrap
```

Use `install_rootfs.sh` script to prepare Root File System. You should have `sudo`

```
$ sudo ./tools/cross/install_rootfs.sh arm
```
- supports `arm`(default) and `aarch` architecutre for now
- supports `xenial`(default) `trusty`, `bionic`, and `focal` release

To see the options,
```
$ ./tools/cross/install_rootfs.sh -h
```

RootFS will be prepared at `tools/cross/rootfs/arm` folder.

***\* CAUTION: The OS version of rootfs must match the OS version of execution target device. On the other hand, you need to match the Ubuntu version of the development PC with the Ubuntu version of rootfs to be used for cross-build. Otherwise, unexpected build errors may occur.***

If you are using Ubuntu 16.04 LTS, select `xenial`, if you are using Ubuntu 18.04 LTS, select `bionic`. You can check your Ubuntu code name in the following way.

```
$ cat /etc/lsb-release
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=18.04
DISTRIB_CODENAME=bionic
DISTRIB_DESCRIPTION="Ubuntu 18.04.4 LTS"
```

If a build error occurs because the version of the development system and the target system do not match, and if you can't replace your development system for any reason, you can consider [cross-build using the docker image](how-to-build-runtime-using-prebuilt-docker-image.md).

### Prepare RootFS at alternative folder

Use `ROOTFS_DIR` to a full path to prepare at alternative path.

```
$ ROOTFS_DIR=/home/user/rootfs/arm-xenial sudo -E ./tools/cross/install_rootfs.sh arm
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

We recommend you have g++ >= 6 installed on your system because NN generated tests require it.

- On Ubuntu 16.04 or older, follow the next steps:

```
$ cd ~/your/path
$ wget https://releases.linaro.org/components/toolchain/binaries/7.2-2017.11/arm-linux-gnueabihf/gcc-linaro-7.2.1-2017.11-x86_64_arm-linux-gnueabihf.tar.xz
$ tar xvf gcc-linaro-7.2.1-2017.11-x86_64_arm-linux-gnueabihf.tar.xz
$ echo 'export PATH=~/your/path/gcc-linaro-7.2.1-2017.11-x86_64_arm-linux-gnueabihf/bin:$PATH' >> ~/.bashrc
```

- On Ubuntu 18.04 LTS, you can install using `apt-get`.
Choose g++ version whatever you prefer: 6, 7, 8 or 9.

```
$ sudo apt-get install g++-{6,7,8,9}-arm-linux-gnueabihf
```

Make sure you get `libstdc++.so` updated on your target with your new toolchain's corresponding one.

For example, if you installed gcc-linaro-7.2.1-2017.11 above, do

```
$ wget https://releases.linaro.org/components/toolchain/binaries/7.2-2017.11/arm-linux-gnueabihf/runtime-gcc-linaro-7.2.1-2017.11-arm-linux-gnueabihf.tar.xz
$ tar xvf runtime-gcc-linaro-7.2.1-2017.11-arm-linux-gnueabihf.tar.xz
```

Then, copy `libstdc++.so.6.0.24` into `/usr/lib/arm-linux-gnueabihf`, and update symbolic links on your device.

## Build and install ARM Compute Library

Mostly you only need once of ACL build.

ACL will be automatically installed in `externals/acl` when you build runtime without any changes.

You can check ACL source information in `cmake/packages/ARMComputeSourceConfig.cmake`

## Cross build for ARM

Give `TARGET_ARCH` variable to set the target architecture.

If you used `ROOTFS_DIR` to prepare in alternative folder, you should also give this to makefile.

```
$ CROSS_BUILD=1 TARGET_ARCH=armv7l make all install

# If ROOTFS_DIR is in alternative folder
$ ROOTFS_DIR=/path/to/your/rootfs/arm \
CROSS_BUILD=1 TARGET_ARCH=armv7l make all install
```

You can also omit the `CROSS_BUILD=1` option if you explicitly pass `ROOTFS_DIR`. In that case, if
the `TARGET_ARCH` are differs from the hostarchitecture, the make script automatically applies
`CROSS_BUILD=1`. So, if you set `ROOTFS_DIR` as an environment variable, you can simply perform
normal build and cross build as follows.

```
$ export ROOTFS_DIR=xxx
...
$ make all install    # do normal build
$ TARGET_ARCH=armv7l make all install    # do cross build
```

### Run test

To run and test the cross-compiled runtime, you need to copy the compiled output to the target device of the architecture in which it is executable.

1. Copy all artifacts under the `./Product` folder to the target device, Odroid-XU4 for example, as a whole.

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

2. Log in to the target device, go to the copied path, and reestore the symbolic link settings of the `Product` directory.

```
$ ssh odroid
sjlee@odroid's password:
...
$ cd ~/one/Product
$ ln ${PWD}/armv7l-linux.debug/obj obj
$ ln ${PWD}/armv7l-linux.debug/out out
$ cd ..
$ ls -la Product
drwxrwxr-x  5 sjlee sjlee 4096 Jun  4 20:55 armv7l-linux.debug
lrwxrwxrwx  1 sjlee sjlee   51 Jun  4 20:54 obj -> /home/sjlee/one/Product/armv7l-linux.debug/obj
lrwxrwxrwx  1 sjlee sjlee   51 Jun  4 20:55 out -> /home/sjlee/one/Product/armv7l-linux.debug/out
```

Now you can test the compilation result in the same way as the native build. Please refer to the following document for details on the test procedure.

- [Testing neural network model inference using the runtime](./how-to-build-runtime.md#run-test)
