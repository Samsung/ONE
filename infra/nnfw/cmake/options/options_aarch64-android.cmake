# aarch64 android cmake options
#
option(BUILD_ANDROID_BENCHMARK_APP "Enable Android Benchmark App" ON)
option(DOWNLOAD_NEON2SSE "Download NEON2SSE library source" OFF)
# Need boost library
option(DOWNLOAD_BOOST "Download boost source" ON)
option(BUILD_BOOST "Build boost source" ON)
option(BUILD_LOGGING "Build logging runtime" OFF)

option(DOWNLOAD_OPENGL_HEADERS "Download Opengl_headers source" ON)
option(DOWNLOAD_EGL_HEADERS "Download Egl_headers source" ON)
option(DOWNLOAD_VULKAN "Download vulkan source" ON)
option(DOWNLOAD_OPENCL_HEADERS "Download Opencl_headers source" ON)

option(BUILD_GPU_CL "Build gpu_cl backend" ON)
option(BUILD_TENSORFLOW_LITE_GPU "Build TensorFlow Lite GPU delegate from the downloaded source" ON)

option(BUILD_MINMAX_H5DUMPER "Build minmax h5dumper" OFF)
option(DOWNLOAD_PYBIND11 "Download Pybind11 source" OFF)
option(BUILD_PYTHON_BINDING "Build python binding" OFF)
