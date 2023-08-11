# How to Build Runtime

This document is based on the system where Ubuntu Desktop Linux 20.04 LTS is installed with default settings, and can be applied in other environments without much difference. For reference, the development of our project started in the Ubuntu Desktop Linux 16.04 LTS environment.

## Build requirements

If you are going to build this project, the following modules must be installed on your system:

- C & C++ compiler
- CMake
- Boost C++ libraries

In the Ubuntu, you can easily install it with the following command.

```
$ sudo apt-get install cmake libboost-all-dev
```

If your linux system does not have the basic development configuration, you will need to install more packages. A list of all packages needed to configure the development environment can be found in https://github.com/Samsung/ONE/blob/master/infra/docker/focal/Dockerfile.

Here is a summary of it for runtime and related tools

```
$ sudo apt install \
build-essential \
clang-format-8 \
cmake \
doxygen \
git \
graphviz \
hdf5-tools \
lcov \
libboost-all-dev \
libhdf5-dev \
python3 \
python3-pip \
scons \
software-properties-common \
unzip \
wget

$ pip3 install yapf==0.22.0 numpy

```

## Build from source code, for Ubuntu

In a typical linux development environment, including Ubuntu, you can build the runtime with a simple command like this:

```
$ git clone https://github.com/Samsung/ONE.git one
$ cd one
$ ./nnfw configure
$ ./nnfw build
$ ./nnfw install
```

For easy build process, we provides `Makefile.template` makefile.

```
$ make -f Makefile.template
```

To release build the runtime, add the environment variable `BUILD_TYPE=release` to the build command as follows.

```
$ export BUILD_TYPE=release
$ make -f Makefile.template
```

Or you can simply do something like this:

```
$ BUILD_TYPE=release make -f Makefile.template
```

The build method described here is a `native build` in which the build environment and execution environment are same. So, this command creates a runtime binary targeting the current build architecture, probably x86_64, as the execution environment. You can find the build output in the ./Product folder as follows:

```
$ tree -L 2 ./Product
./Product
├── out -> /home/sjlee/star/one/Product/x86_64-linux.release/out
└── x86_64-linux.release
    ├── obj
    └── out

5 directories, 3 files

$ tree -L 3 ./Product/out
./Product/out
├── bin
│   ├── onert_run
│   ├── tflite_comparator
│   └── tflite_run
├── include
│   ├── nnfw
│   │   ├── NeuralNetworksEx.h
│   │   ├── NeuralNetworksExtensions.h
│   │   ├── NeuralNetworks.h
│   │   ├── nnfw_experimental.h
│   │   └── nnfw.h
│   └── onert
│       ├── backend
│       ├── compiler
│       ├── exec
│       ├── ir
│       └── util
├── lib
│   ├── libbackend_cpu.so
│   ├── libbackend_ruy.so
│   ├── libneuralnetworks.so
│   ├── libnnfw-dev.so
│   └── libonert_core.so
├── nnapi-gtest
│   ├── nnapi_gtest
│   ├── nnapi_gtest.skip
│   └── nnapi_gtest.skip.x86_64-linux.cpu
├── test
│   ├── command
│   │   ├── nnpkg-test
│   │   ├── prepare-model
│   │   ├── unittest
│   │   └── verify-tflite
│   ├── FillFrom_runner
│   ├── list
│   │   ├── benchmark_nnpkg_model_list.txt
│   │   ├── nnpkg_test_list.armv7l-linux.acl_cl
│   │   ├── nnpkg_test_list.armv7l-linux.acl_neon
│   │   ├── nnpkg_test_list.armv7l-linux.cpu
│   │   ├── tflite_comparator.aarch64.acl_cl.list
│   │   ├── tflite_comparator.aarch64.acl_neon.list
│   │   ├── tflite_comparator.aarch64.cpu.list
│   │   ├── tflite_comparator.armv7l.acl_cl.list
│   │   ├── tflite_comparator.armv7l.acl_neon.list
│   │   ├── tflite_comparator.armv7l.cpu.list
│   │   └── tflite_comparator.x86_64.cpu.list
│   ├── models
│   │   ├── run_test.sh
│   │   └── tflite
│   ├── nnpkgs
│   │   └── FillFrom
│   └── onert-test
└── unittest
    ├── ndarray_test
    ├── nnfw_api_gtest
    ├── nnfw_api_gtest_models
    │   ├── add
    │   ├── add_invalid_manifest
    │   ├── add_no_manifest
    │   ├── if_dynamic
    │   ├── mobilenet_v1_1.0_224
    │   └── while_dynamic
    ├── nnfw_lib_misc_test
    ├── test_cker
    ├── test_onert_core
    ├── test_onert_frontend_nnapi
    └── tflite_test

26 directories, 42 files

```

Here, let's recall that the main target of our project is the arm architecture. If you have a development environment running Linux for arm on a device made of an arm CPU, such as Odroid-XU4, you will get a runtime binary that can be run on the arm architecture with the same command above. This is the simplest way to get a binary for an arm device. However, in most cases, native builds on arm devices are too impractical as they require too long. Therefore, we will create an executable binary of an architecture other than x86_64 through a `cross build`. For cross-build method for each architecture, please refer to the corresponding document in the following section, [How to cross-build runtime for different architecture](#how-to-cross-build-runtime-for-different-architecture).

### Run test

The simple way to check whether the build was successful is to perform inference of the NN model using the runtime. The model to be used for the test can be obtained as follows.

```
$ wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz
$ tar zxvf inception_v3_2018_04_27.tgz ./inception_v3.tflite
$ ls *.tflite
inception_v3.tflite
```

The result of running the inception_v3 model using runtime is as follows. Please consider that this is a test that simply checks execution latency without considering the accuracy of the model.

```
$ ./Product/out/bin/onert_run ./inception_v3.tflite
Model Filename ./inception_v3.tflite
===================================
MODEL_LOAD   takes 1.108 ms
PREPARE      takes 0.190 ms
EXECUTE      takes 183.895 ms
- MEAN     :  183.895 ms
- MAX      :  183.895 ms
- MIN      :  183.895 ms
- GEOMEAN  :  183.895 ms
===================================
```
If you use `tflite_run` instead of `onert_run`, the model will be executed using Tensorflow lite, the basic framework for verification. From the previous build result, you can see that it is the path to the directory where `tflite_run` and `onert_run` are located.

If you come here without any problems, you have all of the basic environments for runtime development.

## Build for Tizen

(Will be written)

## Build using docker image

If your development system is not a linux environment like Ubuntu, but you can use docker on your system, you can build a runtime using a pre-configured docker image. Of course, you can also build a runtime using a docker image in a ubuntu environment, without setting up a complicated development environment. For more information, please refer to the following document.

- [Build using prebuilt docker image](how-to-build-runtime-using-prebuilt-docker-image.md)

## How to cross-build runtime for different architecture

Please refer to the following document for the build method for architecture other than x86_64, which is the basic development environment.

- [Cross building for ARM](how-to-cross-build-runtime-for-arm.md)
- [Cross building for AARCH64](how-to-cross-build-runtime-for-aarch64.md)
- [Cross building for Android](how-to-cross-build-runtime-for-android.md)
