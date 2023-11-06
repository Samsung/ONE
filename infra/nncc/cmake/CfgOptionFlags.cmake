#
# Platform specific configuration
# note: this should be placed before default setting for option setting priority
#       (platform specific setting have higher priority)
#
include("cmake/options/options_${TARGET_PLATFORM}.cmake")

###
### Configuration
###
option(DOWNLOAD_PROTOBUF "Download Protocol Buffer source" ON)
option(BUILD_PROTOBUF "Locally build Protocol Buffer from the downloaded source" ON)
option(DOWNLOAD_EIGEN "Download Eigen source" ON)
option(DOWNLOAD_FARMHASH "Download farmhash source" ON)
option(DOWNLOAD_GEMMLOWP "Download GEMM low precesion library source" ON)
option(DOWNLOAD_RUY "Download ruy source" ON)
option(DOWNLOAD_NEON2SSE "Download NEON2SSE library source" ON)
option(DOWNLOAD_GFLAGS "Download GFlags source" OFF)
option(DOWNLOAD_FLATBUFFERS "Download FlatBuffers source" ON)
option(BUILD_FLATBUFFERS "Locally build Flatbuffers from the downloaded source" ON)
option(DOWNLOAD_TENSORFLOW "Download TensorFlow source" ON)
option(DOWNLOAD_CAFFE "Download Caffe source" ON)
option(DOWNLOAD_PYTORCH "Download Pytorch source" ON)
option(DOWNLOAD_ONNX "Download ONNX source" ON)
option(DOWNLOAD_ABSEIL "Download Abseil-cpp source" ON)
option(DOWNLOAD_OPENCL_HEADERS "Download OpenCl Header source" ON)
option(DOWNLOAD_PYBIND11 "Download Pybind11 source" ON)
option(DOWNLOAD_JSONCPP "Download Jsoncpp source" ON)
option(DOWNLOAD_LIBNPY "Download Libnpy source" ON)

option(DOWNLOAD_GTEST "Download Google Test source" ON)
option(BUILD_GTEST "Build Google Test from the downloaded source" ON)
option(DOWNLOAD_HDF5 "Download HDF5 source" ON)
option(BUILD_HDF5 "Build HDF5 from the downloaded source" ON)

option(ENABLE_STRICT_BUILD "Treat warning as error" OFF)

# This option might be turned ON for Windows native build.
# Check our ProtobufConfig.cmake for its usage.
option(USE_PROTOBUF_LEGACY_IMPORT "Use legacy MODULE mode import rather than CONFIG mode" OFF)

# This option might be turned ON for MCU builds of luci related components.
# It specify which library type to use for build:
# if set ON - luci libraries are static, otherwise - shared.
option(STATIC_LUCI "Build luci as a static libraries" OFF)

# Disable PIC(Position-Independent Code) option for luci-interpreter related components.
# This option might be turned ON for MCU builds.
#
# Enabled PIC requires additional efforts for correct linkage, such as
# implementation of trampoline functions and support of various address tables.
# PIC is used for dynamic libraries, MCU builds of interpreter
# do not benefit from it, so we prefer to disable PIC.
option(NNCC_LIBRARY_NO_PIC "Disable PIC option for libraries" OFF)

# one-cmds PyTorch importer is an experimental feature, it is not used in default configuration.
# This option enables installation of one-import-pytorch utility and
# generation of related testsuite.
option(ENABLE_ONE_IMPORT_PYTORCH "Enable deploy of one-cmds pytoch importer and related tests" OFF)

# Enable exclusion of a module in compiler with exclude.me file
# This option is ignored when BUILD_WHITELIST is given
option(ENABLE_EXCLUDE_ME "Exclude compiler module with exclude.me" ON)
