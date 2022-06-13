# aarch64 android cmake options
#
# NOTE BUILD_ANDROID_TFLITE(JNI lib) is disabled due to BuiltinOpResolver issue.
# tensorflow-lite does not build BuiltinOpResolver but JNI lib need it
# Related Issue : #1403
option(BUILD_ANDROID_TFLITE "Enable android support for TensorFlow Lite" OFF)
option(BUILD_ANDROID_BENCHMARK_APP "Enable Android Benchmark App" ON)
option(DOWNLOAD_NEON2SSE "Download NEON2SSE library source" OFF)
# Need boost library
option(DOWNLOAD_BOOST "Download boost source" ON)
option(BUILD_BOOST "Build boost source" ON)
option(BUILD_LOGGING "Build logging runtime" OFF)
# Do not support npud
option(BUILD_NPUD "Build NPU daemon" OFF)

option(DOWNLOAD_OPENGL_HEADERS "Download Opengl_headers source" ON)
option(DOWNLOAD_EGL_HEADERS "Download Egl_headers source" ON)
option(DOWNLOAD_VULKAN "Download vulkan source" ON)
option(DOWNLOAD_OPENCL_HEADERS "Download Opencl_headers source" ON)
option(DOWNLOAD_PYBIND11 "Download Pybind11 source" ON)
option(BUILD_GPU_CL "Build gpu_cl backend" ON)
option(BUILD_TENSORFLOW_LITE_GPU "Build TensorFlow Lite GPU delegate from the downloaded source" ON)
