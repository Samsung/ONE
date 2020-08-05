# Android Benchmark App

An Android sample app that run `.tflite` and measure performance.

You can run with two engines.

- Tensorflow Lite Interpreter
- NN API Delegate (onert)

## Build

In addition to aarch64-Android build, you need to specify more parameters.

- `ANDROID_BUILD_TOOLS_DIR` : Android `build-tools` directory (You may find it in Android SDK directory)
- `ANDROID_SDK_DIR` : Android SDK directory
- `TFLITE_MODEL_PATH` : A model to run (Only one model can be packed)
- `ANDROID_BOOST_ROOT` : Boost library root path
    - This repo should contain `lib` and `include` directory
    - How to build Boost for Android - Build with [this repo](https://github.com/moritz-wundke/Boost-for-Android)

Example:

```bash
make TARGET_OS=android \
    CROSS_BUILD=1 \
    BUILD_TYPE=RELEASE \
    NDK_DIR=/home/hanjoung/ws/android-tools/r20/ndk \
    EXT_ACL_FOLDER=/home/hanjoung/ws/temp/arm_compute-v19.05-bin-android/lib/android-arm64-v8a-neon-cl \
    ANDROID_BUILD_TOOLS_DIR=/home/hanjoung/ws/android-tools/sdk/build-tools/27.0.3/ \
    ANDROID_SDK_DIR=/home/hanjoung/ws/android-tools/sdk \
    TFLITE_MODEL_PATH=/Users/hanjoung/ws/ghent/STAR/nnfw/tests/scripts/models/cache/MODELS/mobilenet/mobilenet_v1_0.25_128.tflite \
    ANDROID_BOOST_ROOT=/home/hanjoung/ws/gh/moritz-wundke/Boost-for-Android/build/out/arm64-v8a
```

And you will get `obj/contrib/android_benchmark_app/android-benchmark.unsigned.pkg`. This is an unsigned Android app package.

## Sign APK

Before installing the package you probably need to sign the package.

- `apksigner` : This is in `build-tools` directory
- Your keystore : How-to is TBD

```bash
apksigner sign \
    --ks ~/.android/debug.keystore \
    --in Product/aarch64-android.release/obj/contrib/android_benchmark_app/android-benchmark.unsigned.pkg \
    --out tflbench.apk
```

You should enter the keystore password. Then you will get `tflbench.apk`.

## Install APK

```bash
adb install tflbench.apk
adb uninstall com.ndk.tflbench # To uninstall
```
