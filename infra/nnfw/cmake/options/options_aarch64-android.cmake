# aarch64 android cmake options
#
# NOTE BUILD_ANDROID_TFLITE(JNI lib) is disabled due to BuiltinOpResolver issue.
# tensorflow-lite does not build BuiltinOpResolver but JNI lib need it
# Related Issue : #1403
option(BUILD_ANDROID_TFLITE "Enable android support for TensorFlow Lite" ON)
option(BUILD_ANDROID_BENCHMARK_APP "Enable Android Benchmark App" ON)
option(DOWNLOAD_NEON2SSE "Download NEON2SSE library source" OFF)
# Need boost library
option(DOWNLOAD_BOOST "Download boost source" ON)
option(BUILD_BOOST "Build boost source" ON)
option(BUILD_LOGGING "Build logging runtime" OFF)
option(CMAKE_COMPILER_IS_GNUCC "CMAKE_COMPILER_IS_GNUCC" ON)
