# How to Remote Debugging with Visual Studio Code

This document describes how to debug ONE runtime on arm devices using visual studio code.

## Setup build host

### Install gdb-multiarch

1. Install `gdb-multiarch`

```bash
$ sudo apt install gdb-multiarch
```

### Configure VS code

1. Install [Native Debug](https://marketplace.visualstudio.com/items?itemName=webfreak.debug) extension on VS code

2. Setup GDB environment on VS code

- Debug -> Add configuration -> GDB: Connect to gdbserver
- Change configuration as below
  - Change `<TARGET_IP>` to IP of your target
  - The default port number for gdbserver is 2345. You can change this number.
  - You can change `executable` configuration from `tflite_run` to other binaries you want to debug.

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "gdb",
            "request": "attach",
            "name": "Attach to gdbserver",
            "gdbpath": "/usr/bin/gdb-multiarch",
            "executable": "./Product/armv7l-linux.debug/out/bin/tflite_run",
            "target": "<TARGET_IP>:2345",
            "remote": true,
            "printCalls": true,
            "cwd": "${workspaceRoot}",
            "valuesFormatting": "parseText"
        }
    ]
}
```

## Setup target device

### Install gdbserver and debugging symbols

You need to setup a target device for remote debugging.

1. Install `gdbserver`
```bash
$ sudo apt install gdbserver
```

2. Install `libc6-dbg` and copy debugging symbols
```bash
$ sudo apt install libc6-dbg
$ sudo mkdir -p /lib/.debug
$ sudo ln -s /usr/lib/debug/lib/arm-linux-gnueabihf/ld-2.27.so /lib/.debug
```

## Run remote debugging

1. Start gdbserver on target

```bash
gdbserver --multi :<PORT> <BINARY_PATH> <EXECUTION_ARGUMENTS>
```

Example
```bash
gdbserver --multi :2345 Product/armv7l-linux.debug/out/bin/tflite_run ../models/slice_test.tflite
```

2. Connect to gdbserver using VS code
   - Setup breakpoints on any code you want.
   - Click F5 to start remote debugging.
   - Program will execute and exit if no breakpoint exists.

## Optional: Setup rootfs on build host

When debugging starts, `gdb` downloads shared libraries that one runtime uses from the target device.
This process makes `gdb` to wait for shared library download to finish for every debugging start.

To reduce shared library loading, you can setup an arm root file system on your build host and use it.

1. Create arm root file system

Following [CrossBuildForArm](how-to-cross-build-runtime-for-arm.md) to create an arm root file system.

You can use an arm root file system created for arm cross-compile.

2. Install `libc6-dbg` on arm root file system

`<ROOTF_DIR>` should point ARM root file system.

Default path is `tools/cross/rootfs/arm` folder.

```bash
$ sudo chroot <ROOTFS_DIR>
$ apt install libc6-dbg
$ exit
```

3. Create symbolic link of one runtime on arm rootfs

`gdb` will use source code folder at sysroot.

```bash
$ ln -s <ONE_DIR> <ROOTFS_DIR>/<ONE_DIR>
```
Example
```bash
$ ln -s /home/user/one /home/user/one/tools/cross/rootfs/arm/home/user/one/
```

4. Setup `.gdbinit` file on one folder

`gdb` will use `<ROOTFS_DIR>` to find arm related symbols.

```bash
set sysroot <ROOTFS_DIR>
set debug-file-directory <ROOTFS_DIR>/usr/lib/debug
```

## Troubleshooting

### Unable to open 'unordered_map.h'

If you are using docker to build one runtime, you should download and decompress gcc-linaro at `/opt` folder

```bash
wget https://releases.linaro.org/components/toolchain/binaries/6.3-2017.02/arm-linux-gnueabihf/gcc-linaro-6.3.1-2017.02-x86_64_arm-linux-gnueabihf.tar.xz -O gcc-hardfp.tar.xz
sudo tar -xf gcc-hardfp.tar.xz -C /opt/ && sudo rm -rf gcc-hardfp.tar.xz
```

### Skip STL files

Step into (F11) will debug STL files such as `unordered_map` or `vector`.

To skip those files from debugging, you can add below line to `.gdbinit` file.

This function is supported on gdb versions >= 7.12.

```bash
skip -gfile /opt/gcc-linaro-6.3.1-2017.02-x86_64_arm-linux-gnueabihf/arm-linux-gnueabihf/include/c++/6.3.1/bits/*
```
