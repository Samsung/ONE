#
# riscv64 tizen cmake options
#
option(BUILD_ARMCOMPUTE "Build ARM Compute from the downloaded source" OFF)
option(BUILD_TENSORFLOW_LITE "Build TensorFlow Lite from the downloaded source" OFF)
option(DOWNLOAD_ARMCOMPUTE "Download ARM Compute source" OFF)
option(DOWNLOAD_GTEST "Download Google Test source and build Google Test" OFF)

option(GENERATE_RUNTIME_NNAPI_TESTS "Generate NNAPI operation gtest" OFF)
option(ENVVAR_ONERT_CONFIG "Use environment variable for onert configuration" OFF)

option(BUILD_XNNPACK "Build XNNPACK" OFF)

# Tizen boost package does not have static library
option(Boost_USE_STATIC_LIBS "Determine whether or not static linking for Boost" OFF)
