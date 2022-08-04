#
# armv7l tizen cmake options
#
option(BUILD_ARMCOMPUTE "Build ARM Compute from the downloaded source" OFF)
option(BUILD_TENSORFLOW_LITE "Build TensorFlow Lite from the downloaded source" OFF)
option(DOWNLOAD_NEON2SSE "Download NEON2SSE library source" OFF)
option(DOWNLOAD_GTEST "Download Google Test source and build Google Test" OFF)

option(BUILD_LOGGING "Build logging runtime" OFF)
option(GENERATE_RUNTIME_NNAPI_TESTS "Generate NNAPI operation gtest" OFF)
option(ENVVAR_ONERT_CONFIG "Use environment variable for onert configuration" OFF)
option(ENVVAR_NPUD_CONFIG "Use environment variable for npud configuration" OFF)

option(DOWNLOAD_OPENCL_HEADERS "Download Opencl_headers source" ON)
option(DOWNLOAD_TENSORFLOW_GPU "Download Tensorflow GPU delegate source" ON)
option(BUILD_GPU_CL "Build gpu_cl backend" ON)
option(BUILD_TENSORFLOW_LITE_GPU "Build TensorFlow Lite GPU delegate from the downloaded source" ON)
