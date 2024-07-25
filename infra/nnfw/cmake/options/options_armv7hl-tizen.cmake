#
# armv7hl tizen cmake options
#
option(BUILD_ARMCOMPUTE "Build ARM Compute from the downloaded source" OFF)
option(BUILD_TENSORFLOW_LITE "Build TensorFlow Lite from the downloaded source" OFF)
option(DOWNLOAD_ARMCOMPUTE "Build ARM Compute from the downloaded source" OFF)
option(DOWNLOAD_NEON2SSE "Download NEON2SSE library source" OFF)
option(DOWNLOAD_GTEST "Download Google Test source and build Google Test" OFF)

option(BUILD_LOGGING "Build logging runtime" OFF)
option(GENERATE_RUNTIME_NNAPI_TESTS "Generate NNAPI operation gtest" OFF)
option(ENVVAR_ONERT_CONFIG "Use environment variable for onert configuration" OFF)

# Tizen boost package does not have static library
option(Boost_USE_STATIC_LIBS "Determine whether or not static linking for Boost" OFF)
