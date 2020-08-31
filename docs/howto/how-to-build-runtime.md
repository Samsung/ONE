# How to Build Runtime

This document is based on the system where Ubuntu Desktop Linux 18.04 LTS is installed with default settings, and can be applied in other environments without much difference. For reference, the development of our project started in the Ubuntu Desktop Linux 16.04 LTS environment.

## Build requirements

If you are going to build this project, the following modules must be installed on your system:

- CMake
- Boost C++ libraries

In the Ubuntu, you can easily install it with the following command.

```
$ sudo apt-get install cmake libboost-all-dev
```

If your linux system does not have the basic development configuration, you will need to install more packages. A list of all packages needed to configure the development environment can be found in the https://github.com/Samsung/ONE/blob/master/infra/docker/Dockerfile.1804 file.

Here is a summary of it

```
$ sudo apt install \
build-essential \
clang-format-3.9 \
cmake \
doxygen \
git \
graphviz \
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

## Build from source code, for Ubuntu

In a typical linux development environment, including Ubuntu, you can build the runtime with a simple command like this:

```
$ git clone https://github.com/Samsung/ONE.git one
$ cd one
$ make -f Makefile.template install
```

Unfortunately, the debug build on the x86_64 architecture currently has an error. To solve the problem, you must use gcc version 9 or higher. Another workaround is to do a release build rather than a debug build. This is not a suitable method for debugging during development, but it is enough to check the function of the runtime. To release build the runtime, add the environment variable `BUILD_TYPE=release` to the build command as follows.

```
$ export BUILD_TYPE=release
$ make f Makefile.template install
```

Or you can simply do something like this:

```
$ BUILD_TYPE=release make f Makefile.template install
```

The build method described here is a `native build` in which the build environment and execution environment are same. So, this command creates a runtime binary targeting the current build architecture, probably x86_64, as the execution environment. You can find the build output in the ./Product folder as follows:

```
$ tree -L 2 ./Product
./Product
├── obj -> /home/sjlee/star/one/Product/x86_64-linux.release/obj
├── out -> /home/sjlee/star/one/Product/x86_64-linux.release/out
└── x86_64-linux.release
    ├── BUILD
    ├── CONFIGURE
    ├── INSTALL
    ├── obj
    └── out

5 directories, 3 files

$ tree -L 3 ./Product/out
./Product/out
├── bin
│   ├── nnapi_test
│   ├── nnpackage_run
│   ├── tflite_loader_test_tool
│   └── tflite_run
├── include
│   ├── nnfw
│   │   ├── NeuralNetworks.h
│   │   ├── NeuralNetworksEx.h
│   │   ├── NeuralNetworksExtensions.h
│   │   ├── nnfw.h
│   │   └── nnfw_experimental.h
│   └── onert
│       ├── backend
│       ├── compiler
│       ├── exec
│       ├── ir
│       └── util
├── lib
│   ├── libbackend_cpu.so
│   ├── libcircle_loader.so
│   ├── libneuralnetworks.so
│   ├── libnnfw-dev.so
│   ├── libonert_core.so
│   └── libtflite_loader.so
├── test
│   ├── FillFrom_runner
│   ├── command
│   │   ├── nnpkg-test
│   │   ├── prepare-model
│   │   ├── unittest
│   │   └── verify-tflite
│   ├── list
│   │   ├── benchmark_nnpkg_model_list.txt
│   │   ├── frameworktest_list.aarch64.acl_cl.txt
│   │   ├── frameworktest_list.aarch64.acl_neon.txt
│   │   ├── frameworktest_list.aarch64.cpu.txt
│   │   ├── frameworktest_list.armv7l.acl_cl.txt
│   │   ├── frameworktest_list.armv7l.acl_neon.txt
│   │   ├── frameworktest_list.armv7l.cpu.txt
│   │   ├── frameworktest_list.noarch.interp.txt
│   │   ├── frameworktest_list.x86_64.cpu.txt
│   │   ├── nnpkg_test_list.armv7l-linux.acl_cl
│   │   ├── nnpkg_test_list.armv7l-linux.acl_neon
│   │   ├── nnpkg_test_list.armv7l-linux.cpu
│   │   ├── nnpkg_test_list.noarch.interp
│   │   ├── tflite_loader_list.aarch64.txt
│   │   └── tflite_loader_list.armv7l.txt
│   ├── models
│   │   ├── nnfw_api_gtest
│   │   ├── run_test.sh
│   │   └── tflite
│   ├── nnpkgs
│   │   └── FillFrom
│   └── onert-test
├── unittest
│   ├── nnapi_gtest
│   ├── nnapi_gtest.skip
│   ├── nnapi_gtest.skip.noarch.interp
│   └── nnapi_gtest.skip.x86_64-linux.cpu
└── unittest_standalone
    ├── nnfw_api_gtest
    ├── test_compute
    ├── test_onert
    ├── test_onert_backend_cpu_common
    ├── test_onert_frontend_nnapi
    └── tflite_test

20 directories, 47 files

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
$ USE_NNAPI=1 ./Product/out/bin/tflite_run ./inception_v3.tflite
nnapi function 'ANeuralNetworksModel_create' is loaded from '/home/sjlee/star/one/Product/x86_64-linux.release/out/bin/../lib/libneuralnetworks.so'
nnapi function 'ANeuralNetworksModel_addOperand' is loaded from '/home/sjlee/star/one/Product/x86_64-linux.release/out/bin/../lib/libneuralnetworks.so'
nnapi function 'ANeuralNetworksModel_setOperandValue' is loaded from '/home/sjlee/star/one/Product/x86_64-linux.release/out/bin/../lib/libneuralnetworks.so'
nnapi function 'ANeuralNetworksModel_addOperation' is loaded from '/home/sjlee/star/one/Product/x86_64-linux.release/out/bin/../lib/libneuralnetworks.so'
nnapi function 'ANeuralNetworksModel_identifyInputsAndOutputs' is loaded from '/home/sjlee/star/one/Product/x86_64-linux.release/out/bin/../lib/libneuralnetworks.so'
nnapi function 'ANeuralNetworksModel_finish' is loaded from '/home/sjlee/star/one/Product/x86_64-linux.release/out/bin/../lib/libneuralnetworks.so'
nnapi function 'ANeuralNetworksCompilation_create' is loaded from '/home/sjlee/star/one/Product/x86_64-linux.release/out/bin/../lib/libneuralnetworks.so'
nnapi function 'ANeuralNetworksCompilation_finish' is loaded from '/home/sjlee/star/one/Product/x86_64-linux.release/out/bin/../lib/libneuralnetworks.so'
input tensor indices = [317,]
nnapi function 'ANeuralNetworksExecution_create' is loaded from '/home/sjlee/star/one/Product/x86_64-linux.release/out/bin/../lib/libneuralnetworks.so'
nnapi function 'ANeuralNetworksExecution_setInput' is loaded from '/home/sjlee/star/one/Product/x86_64-linux.release/out/bin/../lib/libneuralnetworks.so'
nnapi function 'ANeuralNetworksExecution_setOutput' is loaded from '/home/sjlee/star/one/Product/x86_64-linux.release/out/bin/../lib/libneuralnetworks.so'
nnapi function 'ANeuralNetworksExecution_startCompute' is loaded from '/home/sjlee/star/one/Product/x86_64-linux.release/out/bin/../lib/libneuralnetworks.so'
nnapi function 'ANeuralNetworksEvent_wait' is loaded from '/home/sjlee/star/one/Product/x86_64-linux.release/out/bin/../lib/libneuralnetworks.so'
nnapi function 'ANeuralNetworksEvent_free' is loaded from '/home/sjlee/star/one/Product/x86_64-linux.release/out/bin/../lib/libneuralnetworks.so'
nnapi function 'ANeuralNetworksExecution_free' is loaded from '/home/sjlee/star/one/Product/x86_64-linux.release/out/bin/../lib/libneuralnetworks.so'
output tensor indices = [316(max:905),]
===================================
MODEL_LOAD   takes 1.108 ms
PREPARE      takes 0.190 ms
EXECUTE      takes 183.895 ms
- MEAN     :  183.895 ms
- MAX      :  183.895 ms
- MIN      :  183.895 ms
- GEOMEAN  :  183.895 ms
===================================
nnapi function 'ANeuralNetworksCompilation_free' is loaded from '/home/sjlee/star/one/Product/x86_64-linux.release/out/bin/../lib/libneuralnetworks.so'
nnapi function 'ANeuralNetworksModel_free' is loaded from '/home/sjlee/star/one/Product/x86_64-linux.release/out/bin/../lib/libneuralnetworks.so'
```
Here, `USE_NNAPI=1` means that **ONE** runtime is used for model inference. If omitted, the model will be executed using Tensorflow lite, the basic framework for verification. From the previous build result, you can see that it is the path to the directory where `libneuralnetworks.so` and `libonert_core.so` are located.

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
