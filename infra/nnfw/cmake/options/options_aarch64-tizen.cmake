#
# aarch64 tizen cmake options
#
option(BUILD_ARMCOMPUTE "Build ARM Compute from the downloaded source" OFF)
option(BUILD_TENSORFLOW_LITE "Build TensorFlow Lite from the downloaded source" OFF)
option(DOWNLOAD_EIGEN "Download Eigen source" OFF)
option(DOWNLOAD_NEON2SSE "Download NEON2SSE library source" OFF)

option(BUILD_LOGGING "Build logging runtime" OFF)
option(BUILD_TFLITE_RUN "Build tflite-run" OFF)
option(BUILD_TFLITE_LOADER_TEST_TOOL "Build tflite loader testing tool" OFF)
option(GENERATE_RUNTIME_NNAPI_TESTS "Generate NNAPI operation gtest" OFF)
option(ENVVAR_ONERT_CONFIG "Use environment variable for onert configuration" OFF)
