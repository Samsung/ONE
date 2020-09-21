# Reference: https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/lite/tools/make/Makefile
#
# Tensorflow Lite library 2.3.0
#
set(TENSORFLOW_LITE_BASE ${TFLiteVanillaTensorFlowSource_DIR}/tensorflow/lite)

file(GLOB TFLITE_CORE_SRCS "${TENSORFLOW_LITE_BASE}/*.c"
     "${TENSORFLOW_LITE_BASE}/*.cc"
     "${TENSORFLOW_LITE_BASE}/core/*.cc")

file(GLOB_RECURSE TFLITE_KERNEL_SRCS "${TENSORFLOW_LITE_BASE}/kernels/*.cc")

file(GLOB TFLITE_LIB_SRCS "${TENSORFLOW_LITE_BASE}/c/*.c" "${TENSORFLOW_LITE_BASE}/c/*.cc")

file(GLOB TFLITE_API_SRCS "${TENSORFLOW_LITE_BASE}/core/api/*.c"
     "${TENSORFLOW_LITE_BASE}/core/api/*.cc")

list(APPEND TFLITE_PROFILING_SRCS "${TENSORFLOW_LITE_BASE}/profiling/memory_info.cc")
list(APPEND TFLITE_PROFILING_SRCS "${TENSORFLOW_LITE_BASE}/profiling/time.cc")

file(GLOB TFLITE_EXPERIMENTAL_SRCS "${TENSORFLOW_LITE_BASE}/experimental/resource/*.cc")

file(GLOB TFLITE_SPARSITY_SRCS "${TENSORFLOW_LITE_BASE}/tools/optimize/sparsity/*.cc")

list(APPEND TFLITE_SRCS ${TFLITE_CORE_SRCS})
list(APPEND TFLITE_SRCS ${TFLITE_KERNEL_SRCS})
list(APPEND TFLITE_SRCS ${TFLITE_LIB_SRCS})
list(APPEND TFLITE_SRCS ${TFLITE_API_SRCS})
list(APPEND TFLITE_SRCS ${TFLITE_PROFILING_SRCS})
list(APPEND TFLITE_SRCS ${TFLITE_EXPERIMENTAL_SRCS})
list(APPEND TFLITE_SRCS ${TFLITE_SPARSITY_SRCS})

# externals
list(APPEND TFLITE_SRCS "${TFLiteVanillaFarmhashSource_DIR}/src/farmhash.cc")
list(APPEND TFLITE_SRCS "${TFLiteVanillaFFT2DSource_DIR}/fftsg.c")
list(APPEND TFLITE_SRCS "${TFLiteVanillaFFT2DSource_DIR}/fftsg2d.c")
list(APPEND TFLITE_SRCS "${TFLiteVanillaFlatBuffersSource_DIR}/src/util.cpp")

# externals - absl
file(GLOB_RECURSE ABSL_SRCS "${TFLiteVanillaAbslSource_DIR}/absl/*.cc")
file(GLOB_RECURSE ABSL_EXCLS "${TFLiteVanillaAbslSource_DIR}/absl/*test*.cc"
     "${TFLiteVanillaAbslSource_DIR}/absl/*benchmark*.cc"
     "${TFLiteVanillaAbslSource_DIR}/absl/synchronization/*.cc"
     "${TFLiteVanillaAbslSource_DIR}/absl/debugging/*.cc"
     "${TFLiteVanillaAbslSource_DIR}/absl/hash/*.cc"
     "${TFLiteVanillaAbslSource_DIR}/absl/flags/*.cc"
     "${TFLiteVanillaAbslSource_DIR}/absl/random/*.cc")
list(REMOVE_ITEM ABSL_SRCS ${ABSL_EXCLS})
list(APPEND TFLITE_SRCS ${ABSL_SRCS})

# externals - ruy
file(GLOB RUY_SRCS "${TFLiteVanillaRuySource_DIR}/ruy/*.cc")
file(GLOB_RECURSE RUY_EXCLS "${TFLiteVanillaRuySource_DIR}/ruy/*test*.cc"
      "${TFLiteVanillaRuySource_DIR}/ruy/*benchmark*.cc"
      "${TFLiteVanillaRuySource_DIR}/ruy/*example*.cc")
list(REMOVE_ITEM RUY_SRCS ${RUY_EXCLS})
# Temporary fix for ruy compilation error.
# TODO(b/158800055): Remove this hack once the ruy version is correctly bumped.
list(REMOVE_ITEM RUY_SRCS "${TFLiteVanillaRuySource_DIR}/ruy/prepare_packed_matrices.cc")
list(APPEND TFLITE_SRCS ${RUY_SRCS})


# Build with mmap? true
# caution: v2.3.0's Makefile has wrong code on this part. This is fixed on master branch.
set(BUILD_WITH_MMAP TRUE)
if(${BUILD_WITH_MMAP})
  list(REMOVE_ITEM TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/mmap_allocation_disabled.cc")
else()
  list(REMOVE_ITEM TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/mmap_allocation.cc")
endif()

# Build with nnapi? true
# caution: this nnapi delegate comes from tflite, not ours.
set(BUILD_WITH_NNAPI TRUE)
if(${BUILD_WITH_NNAPI})
  list(APPEND TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/delegates/nnapi/nnapi_delegate.cc")
  list(APPEND TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/delegates/nnapi/quant_lstm_sup.cc")
  list(APPEND TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/nnapi/nnapi_implementation.cc")
  list(APPEND TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/nnapi/nnapi_util.cc")
else()
  list(APPEND TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/delegates/nnapi/nnapi_delegate_disabled.cc")
  list(APPEND TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/nnapi/nnapi_implementation_disabled.cc")
endif()

# ios: we don't support ios
list(REMOVE_ITEM TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/minimal_logging_ios.cc")

# android
if(NOT ANDROID)
  list(REMOVE_ITEM TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/minimal_logging_android.cc")
endif()

# exclude some source files
file(GLOB_RECURSE TFLITE_EXCLS "${TENSORFLOW_LITE_BASE}/*test*.cc"
     "${TENSORFLOW_LITE_BASE}/*benchmark*.cc"
     "${TENSORFLOW_LITE_BASE}/*example*.cc"
     "${TENSORFLOW_LITE_BASE}/*tool*.cc")
list(REMOVE_ITEM TFLITE_SRCS ${TFLITE_EXCLS})

# include headers
list(APPEND TFLITE_INCLUDES "${TFLiteVanillaTensorFlowSource_DIR}")
list(APPEND TFLITE_INCLUDES "${TFLiteVanillaEigenSource_DIR}")
list(APPEND TFLITE_INCLUDES "${TFLiteVanillaAbslSource_DIR}")
list(APPEND TFLITE_INCLUDES "${TFLiteVanillaGEMMLowpSource_DIR}")
list(APPEND TFLITE_INCLUDES "${TFLiteVanillaNEON2SSESource_DIR}")
list(APPEND TFLITE_INCLUDES "${TFLiteVanillaFarmhashSource_DIR}/src")
list(APPEND TFLITE_INCLUDES "${TFLiteVanillaFlatBuffersSource_DIR}/include")
list(APPEND TFLITE_INCLUDES "${TFLiteVanillaFP16Source_DIR}/include")
list(APPEND TFLITE_INCLUDES "${TFLiteVanillaRuySource_DIR}")

add_library(tensorflow-lite-2.3.0 STATIC ${TFLITE_SRCS})
target_include_directories(tensorflow-lite-2.3.0 SYSTEM PUBLIC ${TFLITE_INCLUDES})
target_include_directories(tensorflow-lite-2.3.0 PRIVATE ${CpuInfoSource_DIR})
target_compile_definitions(tensorflow-lite-2.3.0 PUBLIC "GEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK -DTFLITE_WITH_RUY -DTFLITE_WITH_RUY_GEMV -DRUY_HAVE_CPUINFO")
set_property(TARGET tensorflow-lite-2.3.0 PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(tensorflow-lite-2.3.0 eigen ${LIB_PTHREAD} dl cpuinfo)
if(NOT ANDROID AND ${BUILD_WITH_NNAPI})
  target_link_libraries(tensorflow-lite-2.3.0 rt)
endif()

if(ANDROID)
  target_link_libraries(tensorflow-lite-2.3.0 log)
  target_include_directories(tensorflow-lite-2.3.0 PUBLIC "${NDK_DIR}/..")
endif()
