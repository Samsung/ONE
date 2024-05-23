#
# armv7l linux cmake options
#
option(DOWNLOAD_NEON2SSE "Download NEON2SSE library source" OFF)
option(BUILD_OPENCL_TOOL "Build OpenCL tool" ON)

option(DOWNLOAD_PYBIND11 "Download Pybind11 source" ON)
option(BUILD_PYTHON_BINDING "Build python binding" ON)

# Under linux gcc 10.0, required header for xnnpack arm build is not supported
cmake_dependent_option(BUILD_XNNPACK "Build xnnpack library from the downloaded source" OFF "CXX_COMPILER_VERSION VERSION_LESS 10.0" ON)
