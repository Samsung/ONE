# onert-micro

`onert-micro`(a.k.a `luci-micro`) is MCU specialized build of luci-interpreter with several benchmark applications.

## Contents

onert-micro contains cmake infrastructure to build:
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
$ cmake ../infra/onert-micro
$ make -j$(nproc) luci_interpreter_micro_arm
```

### Known issues

Interpreter uses TensorFlow headers that produces warnings.

`Linux` x86 build uses "-isystem" flag to suppress warnings from external sources,
but some old arm compilers have issues with it:
[bug](https://bugs.launchpad.net/gcc-arm-embedded/+bug/1698539)

`-isystem` hack is disabled for MCU build, because of this MCU build is broken if `-Werror` flag is set.

## How to use

### Convert tflite model to circle model

To inference with tflite model, you need to convert it to circle model format(https://github.com/Samsung/ONE/blob/master/res/CircleSchema/0.4/circle_schema.fbs).
Please refer to `tflite2circle` tool(https://github.com/Samsung/ONE/tree/master/compiler/tflite2circle) for this purpose.

### Convert to c array model

Many MCU platforms are lack of file system support. The proper way to provide a model to onert-micro is to convert it into c array so that it can be compiled into MCU binary. 

``` bash
xxi -i model.circle > model.h
```

Then, model.h looks like this: 

``` cpp
unsigned char model_circle[] = {
  0x22, 0x01, 0x00, 0x00, 0xf0, 0x00, 0x0e, 0x00,
  // .....
};
unsigned int model_circle_len = 1004;
```

### API

Once you have c array model, you are ready to use onert-micro.

To run a model with onert-micro, follow the instruction: 

1. Include onert-micro header

``` cpp
#include <luci_interpreter/Interpreter.h>
```

2. Create interpreter instance

onert-micro interpreter expects model as c array as mentioned in [Previous Section](#convert-to-c-array-model).

``` cpp
#include "model.h"

luci_interpreter::Interpreter interpreter(model_circle, true);
```

3. Feed input data

To feed input data into interpreter, we need to do two steps: 1) allocate input tensors and 2) copy input into input tensors.

``` cpp
    for (int32_t i = 0; i < num_inputs; i++)
    {
      auto input_data = reinterpret_cast<char *>(interpreter.allocateInputTensor(i));
      readDataFromFile(std::string(input_prefix) + std::to_string(i), input_data,
                       interpreter.getInputDataSizeByIndex(i));
    }
```

4. Do inference

``` cpp
    interpreter.interpret();
```

5. Get output data

``` cpp
    auto data = interpreter.readOutputTensor(i);
```


### Reduce Binary Size

onert-micro provides compile flags to generate reduced-size binary.

- `DIS_QUANT` : Flag for Disabling Quantized Type Operation
- `DIS_FLOAT` : Flag for Disabling Float Operation
- `DIS_DYN_SHAPES` : Flag for Disabling Dynamic Shape Support

Also, you can build onert-micro library only with kernels in target models.
For this, please remove all the kernels from [KernelsToBuild.lst](./luci-interpreter/pal/mcu/KernelsToBuild.lst) except kernels in your target model. 
