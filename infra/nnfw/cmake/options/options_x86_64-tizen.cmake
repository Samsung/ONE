#
# x86_64 linux cmake options
#
option(BUILD_ARMCOMPUTE "Build ARM Compute from the downloaded source" OFF)
option(BUILD_TENSORFLOW_LITE "Build TensorFlow Lite from the downloaded source" OFF)
option(DOWNLOAD_ARMCOMPUTE "Download ARM Compute source" OFF)
option(DOWNLOAD_GTEST "Download Google Test source and build Google Test" OFF)

option(BUILD_LOGGING "Build logging runtime" OFF)
option(GENERATE_RUNTIME_NNAPI_TESTS "Generate NNAPI operation gtest" OFF)
option(ENVVAR_ONERT_CONFIG "Use environment variable for onert configuration" OFF)

option(BUILD_XNNPACK "Build XNNPACK" OFF)
option(DOWNLOAD_OPENCL_HEADERS "Download opencl headers" OFF)

option(BUILD_NPUD "Build NPU daemon" ON)
# Do not allow to use CONFIG option on Tizen
option(ENVVAR_NPUD_CONFIG "Use environment variable for npud configuration" OFF)

option(BUILD_MINMAX_H5DUMPER "Build minmax h5dumper" OFF)
