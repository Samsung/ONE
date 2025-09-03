# How to Cross-build Runtime for Android

**Note: To set up a basic build environment on your development PC, please read the [how-to-build-runtime.md](how-to-build-runtime.md) document first. The cross build covered in this document assumes that you have an environment in which the native build operates normally without problems.**

Supported Architecture : AARCH64 only (ARM32 is not supported yet)

## Prepare Android NDK

There are two ways to get Android NDK: install Android NDK on your PC or use docker image which already has Android NDK installed.

### Install Android NDK

Use `tools/cross/install_android_ndk.sh` script to prepare Android NDK. This is recommended way to build Android NDK.

Or you can use `tools/cross/install_android_sdk.sh` script to prepare Android SDK including NDK. You can find NDK in `{android-sdk-dir}/ndk/{ndk-version}` directory.

CMake 3.6.0 or later is required for Android NDK r26d CMake support.

### Host Environment Requirements

If you want to use docker image, please use `infra/docker/android-sdk/Dockerfile` of our repository or pull the image `nnfw/one-devtools:android-sdk` from `DockerHub`. It includes Android SDK and NDK.

You can build the docker image with following command:
```
$ docker build -t nnfw/one-devtools:android-sdk -f infra/docker/android-sdk/Dockerfile .
```

## Build

### Build and install the runtime

Some tools/libs are still not supported and those are not built by default - mostly due to dependency on HDF5 library.

Different from cross build for linux,

- Wrapper toolchain file for Android NDK: `runtime/infra/cmake/buildtool/cross/toolchain_aarch64-android.cmake`
- `ANDROID_NDK` environment variable or `CMAKE_ANDROID_NDK` cmake variable is required

If you are using docker image `nnfw/one-devtools:android-sdk`, you don't need to specify above variable because environment variable is already set in the image.

Here is an example of using Makefile.

```bash
TARGET_OS=android \
CROSS_BUILD=1 \
ANDROID_NDK=/path/android-sdk/ndk/{ndk-version}/ \
make -f Makefile.template install
```

Otherwise, you can use command
```
$ source .venv/bin/activate
(.venv)$ ./nnfw configure \
  -DCMAKE_TOOLCHAIN_FILE=infra/cmake/buildtool/cross/toolchain_aarch64-android.cmake \
  -DCMAKE_ANDROID_NDK=/path/android-sdk/ndk/{ndk-version}
(.venv)$ ./nnfw build
(.venv)$ ./nnfw install --prefix build/onert/out
```
