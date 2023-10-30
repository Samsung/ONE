## How to build an example
- Build ONE toolchain
- Convert your tflite model to circle using tflite2circle from toolchain
  e.g. `./tflite2circle ./speech_recognition_float.tflite ./speech_recognition_float.circle`
- Convert model to C array: `xxd -i ./speech_recognition_float.circle ./speech_recognition_float.h`
- Include C array with model to your project
