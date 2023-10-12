# Cross building for Android

**Note: To set up a basic build environment on your development PC, please read the [howto.md](../howto.md) document first. The cross build covered in this document assumes that you have an environment in which the native build operates normally without problems.**

Supported Architecture : AARCH64 only (ARM32 is not supported yet)

## Prepare Android NDK

Use `tools/cross/install_android_sdk.sh` script to prepare Android NDK. This is recommended way to build Android NDK.
You may download it yourself from the offical Android NDK website, but the script does a little more than just downloading and unzipping.

```
$ ./tools/cross/install_android_sdk.sh --ndk-only

Android NDK is installed on /{project_root}/tools/cross/android_sdk/ndk/{ndk_version}
```

## Build

### Host Environment Requirements

With Ubuntu 20.04, everything is fine except one. CMake 3.6.0 or later is required for Android NDK CMake support.
So if you want to use Docker, please use `infra/docker/focal/Dockerfile` which is based on Ubuntu 20.04. It has CMake 3.16.3.

```
$ ./nnas build-docker-image -t nnfw/one-devtools:focal
```

### Get prebuilt ARM Compute Library

Download prebuilt binary from [github](https://github.com/ARM-software/ComputeLibrary/releases). Check the version we support and platform(Android).

Then extract the tarball to the folder indicated as EXT_ACL_FOLDER in the example below. We will use the following file in `lib/android-arm64-v8a-neon-cl`.

```
libarm_compute_core.so
libarm_compute_graph.so
libarm_compute.so
```

### Build and install the runtime

Some tools/libs are still not supported and those are not built by default - mostly due to dependency on Boost library.
Please refer to `infra/nnfw/cmake/options/options_aarch64-android.cmake` for details.

Different from cross build for linux,

- `NDK_DIR` is required

Here is an example of using Makefile.

```bash
TARGET_OS=android \
CROSS_BUILD=1 \
NDK_DIR=/path/android-sdk/ndk/{ndk-version}/ \
EXT_ACL_FOLDER=/path/arm_compute-v19.11.1-bin-android/lib/android-arm64-v8a-neon-cl \
make -f Makefile.template install
```
