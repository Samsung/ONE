include(CMakeDependentOption)

#
# Platfor specific configuration
# note: this should be placed before default setting for option setting priority
#       (platform specific setting have higher priority)
#
include("${CMAKE_CURRENT_LIST_DIR}/options/options_${TARGET_PLATFORM}.cmake" OPTIONAL)

#
# Default build configuration for project
#
option(ENABLE_STRICT_BUILD "Treat warning as error" ON)
option(ENABLE_COVERAGE "Build for coverage test" OFF)
option(ASAN_BUILD "Address Sanitizer build" OFF)

#
# Default build configuration for runtime
#
option(ENVVAR_ONERT_CONFIG "Use environment variable for onert configuration" ON)

#
# Default build configuration for tests
#
option(BUILD_RUNTIME_NNAPI_TEST "Build Runtime NN API Generated Test" ON)
option(BUILD_RUNTIME_NNFW_API_TEST "Build Runtime NNFW API Tests" ON)
option(BUILD_TFLITE_RUN "Build tflite_run test driver" ON)
option(BUILD_ONERT_RUN "Build onert_run test driver" ON)
option(BUILD_ONERT_TRAIN "Build onert_train test driver" ON)
option(BUILD_TFLITE_COMPARATOR_TEST_TOOL "Build testing tool to compare runtime result with TFLite" ON)
option(BUILD_WITH_HDF5 "Build test tool with HDF5 library" ON)
option(GENERATE_RUNTIME_NNAPI_TESTS "Generate NNAPI operation gtest" ON)

#
# Default build configuration for contrib
#
option(BUILD_ANDROID_BENCHMARK_APP "Enable Android Benchmark App" OFF)
option(BUILD_DETECTION_APP "Build detection example app" OFF)
option(BUILD_HEAP_TRACE "Build heap trace tool" OFF)
option(BUILD_LABS "Build lab projects" OFF)
option(BUILD_STYLE_TRANSFER_APP "Build style transfer app" OFF)
option(BUILD_TFLITE_TEST "Build tensorflow lite test" OFF)
option(BUILD_TFLITE_CLASSIFY_APP "Build tflite_classify app" OFF)
option(BUILD_UBEN "Build micro-benchmark (uben) suite" OFF)
option(BUILD_MLAPSE "Build mlapse benchmark toolkit" OFF)
option(BUILD_GPU_CL "Build gpu_cl backend" OFF)
option(BUILD_TENSORFLOW_LITE_GPU "Build TensorFlow Lite GPU delegate from the downloaded source" OFF)
option(BUILD_NPUD "Build NPU daemon" OFF)
option(ENVVAR_NPUD_CONFIG "Use environment variable for npud configuration" OFF)
option(BUILD_LOGGING "Build logging runtime" OFF)
option(BUILD_XNNPACK_BACKEND "Build XNNPack backend" OFF)

#
# Default build configuration for tools
#
option(BUILD_KBENCHMARK "Build kernel benchmark tool" OFF)
option(BUILD_OPENCL_TOOL "Build OpenCL tool" OFF)

#
# Default external libraries source download and build configuration
#
option(DOWNLOAD_TENSORFLOW "Download Tensorflow source" ON)
option(DOWNLOAD_ABSEIL "Download Abseil source" ON)
option(DOWNLOAD_EIGEN "Download Eigen source" ON)
option(DOWNLOAD_FARMHASH "Download farmhash source" ON)
option(DOWNLOAD_GEMMLOWP "Download GEMM low precesion library source" ON)
option(DOWNLOAD_NEON2SSE "Download NEON2SSE library source" ON)
option(DOWNLOAD_FLATBUFFERS "Download FlatBuffers source" ON)
option(DOWNLOAD_ARMCOMPUTE "Download ARM Compute source" OFF)
option(DOWNLOAD_NONIUS "Download nonius source" ON)
option(DOWNLOAD_RUY "Download ruy source" ON)
option(DOWNLOAD_CPUINFO "Download cpuinfo source" ON)
option(DOWNLOAD_OOURAFFT "Download Ooura FFT source" ON)
option(DOWNLOAD_MLDTYPES "Download ml_dtypes source" ON)
option(DOWNLOAD_GTEST "Download Google Test source and build Google Test" ON)
option(BUILD_TENSORFLOW_LITE "Build TensorFlow Lite from the downloaded source" ON)
option(BUILD_ARMCOMPUTE "Build ARM Compute from the downloaded source" OFF)
option(DEBUG_ARMCOMPUTE "Build ARM Compute as debug type" OFF)
option(BUILD_RUY "Build ruy library from the downloaded source" ON)
option(BUILD_CPUINFO "Build cpuinfo library from the downloaded source" ON)
option(PROFILE_RUY "Enable ruy library profiling" OFF)
option(DOWNLOAD_XNNPACK "Download xnnpack source" ON)
option(BUILD_XNNPACK "Build xnnpack library from the downloaded source" ON)
option(DOWNLOAD_PTHREADPOOL "Download pthreadpool source" ON)
option(BUILD_PTHREADPOOL "Build pthreadpool library from the source" ON)
option(DOWNLOAD_PSIMD "Download psimd source" ON)
option(BUILD_PSIMD "Build psimd library from the source" ON)
option(DOWNLOAD_FP16 "Download fp16 source" ON)
option(BUILD_FP16 "Build fp16 library from the source" ON)
option(DOWNLOAD_FXDIV "Download fxdiv source" ON)
option(BUILD_FXDIV "Build fxdiv library from the source" ON)
option(DOWNLOAD_PYBIND11 "Download Pybind11 source" OFF)
option(BUILD_PYTHON_BINDING "Build python binding" OFF)
option(HDF5_USE_STATIC_LIBRARIES "Determine whether or not static linking for HDF5" ON)

#
## Default sample build configuration
#
option(BUILD_MINIMAL_SAMPLE "Build minimal app" ON)
