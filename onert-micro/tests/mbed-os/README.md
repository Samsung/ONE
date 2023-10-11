## Typical use
- Build ONE toolchain
- Convert your tflite model to circle using tflite2circle from toolchain
 e.g. `./tflite2circle ./float_model_speech.tflite ./float_model_speech.circle`
- Convert model to C array: `xxd -i ./float_model_speech.circle ./float_model_speech.h`
- Include C array to your project

## How to build an example
- From repository root `mkdir build_onert-micro`
- `cd build_onert-micro`
- `cmake ../infra/onert-micro/ -DBUILD_TEST=1`
- `make -j$(nproc) onert_micro_build_test_arm`
- Flash `build_onert-micro/onert-micro/tests/mbed-os/build_test.bin` to MCU