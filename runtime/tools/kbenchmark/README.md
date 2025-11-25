# kbenchmark

## Purpose

Most people can use several neural network kernel libraries to inference AI models on Linux system. Each kernel library is implemented in a number of ways, which can give different performance to different configurations. This benchmark tool can help you measure the system's performance with each configuration, and identify possible issues on your system.

## Prerequisites

### Relevant libraries
This script is based on [libnonius](https://github.com/libnonius/nonius) micro-benchmarking framework. This micro-benchmarking framework will be downloaded automatically if you set the `BUILD_KBENCHMARK` configuration to on and build nnfw. If you want to download manually, please use the following path.
* [Download Nonius header files](https://github.com/libnonius/nonius/releases)

### Configuration file
This tool works depending on a configuration file generated from [summarize_tflite.sh](nnfw/blob/master/tools/tflkit/summarize_tflite.sh). You can make a configuration file using the following commands.
```
nnfw$ cd tools/tflkit
nnfw/tools/tflkit$ ./summarize_tflite.sh [tflite model file] -c -p [file prefix]
```
Or
```
nnfw$ cd tools/tflitefile_tool
nnfw/tools/tflitefile_tool$ python model_parser.py [tflite model file] -c -p [file prefix]
```

The generated configuration file will have the following format:
```
tools/tflitefile_tool$ cat inceptionv3_slim_Main_model_CONV_2D.config | head -n 25
# CONV_2D, Total count: 95

[0]
input: [1, 299, 299, 3]
input_type: FLOAT32
weights: [32, 3, 3, 3]
weights_type: FLOAT32
bias: [32]
bias_type: FLOAT32
output_counts: 1
output0: [1, 149, 149, 32]
output0_type: FLOAT32
stride_w: 2
stride_h: 2
dilation_w: 1
dilation_h: 1
padding: VALID
fused_act: RELU

[1]
input: [1, 149, 149, 32]
input_type: FLOAT32
weights: [32, 3, 3, 32]
weights_type: FLOAT32
bias: [32]
```

### Benchmark kernel library
This tool needs kernel benchmark libraries. The kernel benchmark library depends on `nonius` c++ micro-benchmarking framework. You can get the detail guideline in [libnonius/nonius](https://github.com/libnonius/nonius) github repository. The `nonius` library uses morden C++ and is header only. The kernel benchmark libraries will be linked to `kbenchmark` tool using dynamic linking loader. So, it should export the `nonius::benchmark_registry &benchmark_functions(void)` symbol. This symbol should return the nonius benchmark test lists. You can see all benchmark test lists that are executed using `--verbose` option as log.

## kbenchmark

### Available commands
The binary takes the following required parameters:

* `config`: `string` \
  The path to the configuration file.
* `kernel`: `string` \
  The path to the benchmark kernel library file. It allows multiple kernel libraries either by using space or by repeatedly calling `--kernel`.

and the following optional parameters:

* `reporter`: `string` \
  Set the reporter types among `standard`, `html`, `junit` or `csv`. Default reporter type is `standard`.
* `output`: `string` \
  Set the additional strings for output file name.
* `help`: \
  Display available options.
* `verbose`: \
  Show verbose messages.

### Operations
The `OperationLoader` loads each operation information from configuration file. This loader takes the last string of the configuration file name as a key of `OperationLoader` map. So the configuration file should not be changed. For example, if the configuration file name is a `inceptionv3_slim_Main_model_CONV_2D.test.config`, the `OperationLoader` takes `CONV_2D` as a key of map. The `CONV_2D` key is connected to `Convolution` class in `operations/Convolution.h`. This related information is described in `Operations.lst` file. Each operation class will return the `nonius::parameters` from `OperationInfo` in `ConfigFile` class.
