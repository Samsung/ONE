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

option(DOWNLOAD_GTEST "Download Google Test source" ON)
option(BUILD_GTEST "Build Google Test from the downloaded source" ON)
