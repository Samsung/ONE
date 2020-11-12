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
option(BUILD_RUNTIME_NNAPI_TEST "Build Runtime NN API Generated Test" ON)
option(BUILD_NNAPI_TEST "Build nnapi_test" ON)
option(BUILD_NNPACKAGE_RUN "Build nnpackge_run" ON)
option(BUILD_TFLITE_RUN "Build tflite-run" ON)
option(BUILD_TFLITE_LOADER_TEST_TOOL "Build tflite loader testing tool" ON)
option(BUILD_LOGGING "Build logging runtime" OFF)
