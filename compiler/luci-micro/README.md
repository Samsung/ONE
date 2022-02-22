# luci-micro

`luci-micro` is MCU specialized build of luci-interpreter with several benchmark applications.

## Contents

Luci-micro contains cmake infrastructure to build:
- stand-alone interpreter library
- benchmark applications using luci interpreter on arm MCUs

## How to build stand alone library

Stand-alone library is simply built by `luci_interpreter_micro_arm` target.
Result library will be placed in  `<ONE root>/build/compiler/luci-micro/standalone_arm/luci-interpreter/src/libluci_interpreter.a`.

### Prerequisites

- Everything you need for ONE project: see [how-to-build-compiler.md](../../docs/howto/how-to-build-compiler.md)
- arm-none-eabi-gcc and arm-none-eabi-g++ compilers

To install needed arm compilers on ubuntu:
```
$ sudo apt-get install gcc-arm-none-eabi
```

**cmake build**

``` bash
$ cd <path to ONE>
$ mkdir build
# cd build
$ cmake ../infra/nncc
$ make -j$(nproc) luci_interpreter_micro_arm
```

**nncc script build**

``` bash
$ cd <path to ONE>
$ ./nncc configure
$ ./nncc build -j$(nproc) luci_interpreter_micro_arm
```

### Known issues

Interpreter uses TensorFlow headers that produces warnings.

`Linux` x86 build uses "-isystem" flag to suppress warnings from external sources,
but some old arm compilers have issues with it:
[bug](https://bugs.launchpad.net/gcc-arm-embedded/+bug/1698539)

`-isystem` hack is disabled for MCU build, because of this MCU build is broken if `-Werror` flag is set.

## How to use

TBD
