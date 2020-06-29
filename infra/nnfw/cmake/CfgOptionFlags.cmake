include(CMakeDependentOption)

#
# Platfor specific configuration
# note: this should be placed before default setting for option setting priority
#       (platform specific setting have higher priority)
#
include("cmake/options/options_${TARGET_PLATFORM}.cmake")

#
# Default build configuration for project
#
option(ENABLE_STRICT_BUILD "Treat warning as error" ON)
option(ENABLE_COVERAGE "Build for coverage test" OFF)
option(BUILD_EXT_MULTITHREAD "Build external build using multi thread" ON)
option(BUILD_ONERT "Build onert" ON)
option(BUILD_LOGGING "Build logging runtime" ON)
CMAKE_DEPENDENT_OPTION(BUILD_RUNTIME_NNAPI_TEST "Build Runtime NN API Generated Test"
                       # Set BUILD_RUNTIME_NNAPI_TEST as ON
                       #   if CMAKE_COMPILER_IS_GNUCC AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.2
                       ON "CMAKE_COMPILER_IS_GNUCC;NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.2"
                       # Otherwise set BUILD_RUNTIME_NNAPI_TEST as OFF
                       OFF)
option(BUILD_RUNTIME_NNFW_API_TEST "Build Runtime NNFW API Tests" ON)
option(BUILD_TFLITE_RUN "Build tflite-run" ON)
option(BUILD_TFLITE_RUN_2_2_0 "Build tflite-run 2.2.0" OFF)
option(BUILD_TFLITE_BENCHMARK_MODEL "Build tflite benchmark model" OFF)
option(BUILD_NNPACKAGE_RUN "Build nnpackge_run" ON)
option(BUILD_TFLITE_LOADER "Build TensorFlow Lite loader" ON)
option(BUILD_CIRCLE_LOADER "Build circle loader" ON)
option(BUILD_TFLITE_LOADER_TEST_TOOL "Build tflite loader testing tool" ON)
option(GENERATE_RUNTIME_NNAPI_TESTS "Generate NNAPI operation gtest" ON)
option(ENVVAR_ONERT_CONFIG "Use environment variable for onert configuration" ON)
option(INSTALL_TEST_SCRIPTS "Install test scripts" ON)
#
# Default build configuration for contrib
#
option(BUILD_ANDROID_TFLITE "Enable android support for TensorFlow Lite" OFF)
option(BUILD_ANDROID_BENCHMARK_APP "Enable Android Benchmark App" OFF)
option(BUILD_BENCHMARK_ACL "Build ARM Compute Library Benchmarks" OFF)
option(BUILD_DETECTION_APP "Build detection example app" OFF)
option(BUILD_HEAP_TRACE "Build heap trace tool" OFF)
option(BUILD_LABS "Build lab projects" OFF)
option(BUILD_STYLE_TRANSFER_APP "Build style transfer app" OFF)
option(BUILD_TFLITE_TEST "Build tensorflow lite test" OFF)
option(BUILD_TFLITE_CLASSIFY_APP "Build tflite_classify app" OFF)
option(BUILD_UBEN "Build micro-benchmark (uben) suite" OFF)
option(BUILD_MLAPSE "Build mlapse benchmark toolkit" OFF)
#
# Default build configuration for tools
#
option(BUILD_KBENCHMARK "Build kernel benchmark tool" OFF)
option(BUILD_OPENCL_TOOL "Build OpenCL tool" OFF)
option(BUILD_NNAPI_QUICKCHECK "Build NN API Quickcheck tools" OFF)
option(BUILD_TFLITE_ACCURACY "Build tflite accuracy tool" OFF)
#
# Default external libraries source download and build configuration
#
option(DOWNLOAD_TENSORFLOW "Download Tensorflow source" ON)
option(DOWNLOAD_ABSL "Download Absl source" ON)
option(DOWNLOAD_EIGEN "Download Eigen source" ON)
option(DOWNLOAD_FARMHASH "Download farmhash source" ON)
option(DOWNLOAD_GEMMLOWP "Download GEMM low precesion library source" ON)
option(DOWNLOAD_NEON2SSE "Download NEON2SSE library source" ON)
option(DOWNLOAD_FLATBUFFERS "Download FlatBuffers source" ON)
option(DOWNLOAD_ARMCOMPUTE "Download ARM Compute source" ON)
option(DOWNLOAD_NONIUS "Download nonius source" ON)
option(DOWNLOAD_BOOST "Download boost source" OFF)
option(DOWNLOAD_RUY "Download ruy source" ON)
option(BUILD_BOOST "Build boost source" OFF)
option(BUILD_TENSORFLOW_LITE "Build TensorFlow Lite from the downloaded source" ON)
option(BUILD_TENSORFLOW_LITE_2_2_0 "Build TensorFlow Lite from the downloaded source" OFF)
option(BUILD_GTEST "Download and build Google Test" ON)
option(BUILD_ARMCOMPUTE "Build ARM Compute from the downloaded source" ON)
option(BUILD_RUY "Build ruy library from the downloaded source" ON)
option(PROFILE_RUY "Enable ruy library profiling" OFF)

#
## Default sample build configuration
#
option(BUILD_MINIMAL_SAMPLE "Build minimal app" OFF)
