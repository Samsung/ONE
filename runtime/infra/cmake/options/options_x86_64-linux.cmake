#
# x86_64 linux cmake options
#
option(BUILD_XNNPACK "Build XNNPACK" OFF)
option(DOWNLOAD_PYBIND11 "Download Pybind11 source" ON)
option(BUILD_PYTHON_BINDING "Build python binding" ON)
# Assume x86-64 linux trix backend is used on TRIX simulator
set(TRIX_REQ_TIMEOUT_SEC 180 CACHE STRING "Timeout for TRIX request")
