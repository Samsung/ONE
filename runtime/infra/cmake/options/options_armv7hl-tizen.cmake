#
# armv7hl tizen cmake options
#
option(BUILD_TENSORFLOW_LITE "Build TensorFlow Lite from the downloaded source" OFF)
option(DOWNLOAD_NEON2SSE "Download NEON2SSE library source" OFF)
option(DOWNLOAD_GTEST "Download Google Test source and build Google Test" OFF)

option(ENVVAR_ONERT_CONFIG "Use environment variable for onert configuration" OFF)

option(BUILD_XNNPACK "Build XNNPACK" OFF)
