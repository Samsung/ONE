#
# armv7l linux cmake options
#
option(DOWNLOAD_NEON2SSE "Download NEON2SSE library source" OFF)
option(BUILD_OPENCL_TOOL "Build OpenCL tool" ON)

option(DOWNLOAD_OPENGL_HEADERS "Download Opengl_headers source" ON)
option(DOWNLOAD_EGL_HEADERS "Download Egl_headers source" ON)
option(DOWNLOAD_VULKAN "Download vulkan source" ON)
option(DOWNLOAD_OPENCL_HEADERS "Download Opencl_headers source" ON)
option(BUILD_GPU_CL "Build gpu_cl backend" ON)
option(BUILD_TENSORFLOW_LITE_GPU "Build TensorFlow Lite GPU delegate from the downloaded source" ON)
option(DOWNLOAD_PYBIND11 "Download Pybind11 source" ON)
option(BUILD_PYTHON_BINDING "Build python binding" ON)
