# Reference: https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/lite/tools/make/Makefile
#
# Tensorflow Lite library 2.2.0
#
set(TENSORFLOW_LITE_BASE ${TFLite220TensorFlowSource_DIR}/tensorflow/lite)

file(GLOB TFLITE_CORE_SRCS "${TENSORFLOW_LITE_BASE}/*.c"
     "${TENSORFLOW_LITE_BASE}/*.cc"
     "${TENSORFLOW_LITE_BASE}/core/*.cc")

file(GLOB_RECURSE TFLITE_KERNEL_SRCS "${TENSORFLOW_LITE_BASE}/kernels/*.cc")

file(GLOB TFLITE_LIB_SRCS "${TENSORFLOW_LITE_BASE}/c/*.c" "${TENSORFLOW_LITE_BASE}/c/*.cc")

file(GLOB TFLITE_API_SRCS "${TENSORFLOW_LITE_BASE}/core/api/*.c"
     "${TENSORFLOW_LITE_BASE}/core/api/*.cc")

list(APPEND TFLITE_PROFILING_SRCS "${TENSORFLOW_LITE_BASE}/profiling/memory_info.cc")
list(APPEND TFLITE_PROFILING_SRCS "${TENSORFLOW_LITE_BASE}/profiling/time.cc")

file(GLOB TFLITE_EXPERIMENTAL_SRCS "${TENSORFLOW_LITE_BASE}/experimental/resource/*.cc"
     "${TENSORFLOW_LITE_BASE}/experimental/ruy/*.cc")

file(GLOB TFLITE_SPARSITY_SRCS "${TENSORFLOW_LITE_BASE}/tools/optimize/sparsity/*.cc")

list(APPEND TFLITE_SRCS ${TFLITE_CORE_SRCS})
list(APPEND TFLITE_SRCS ${TFLITE_KERNEL_SRCS})
list(APPEND TFLITE_SRCS ${TFLITE_LIB_SRCS})
list(APPEND TFLITE_SRCS ${TFLITE_API_SRCS})
list(APPEND TFLITE_SRCS ${TFLITE_PROFILING_SRCS})
list(APPEND TFLITE_SRCS ${TFLITE_EXPERIMENTAL_SRCS})
list(APPEND TFLITE_SRCS ${TFLITE_SPARSITY_SRCS})

# externals
list(APPEND TFLITE_SRCS "${TFLite220FarmhashSource_DIR}/src/farmhash.cc")
list(APPEND TFLITE_SRCS "${TFLite220FFT2DSource_DIR}/fftsg.c")
list(APPEND TFLITE_SRCS "${TFLite220FFT2DSource_DIR}/fftsg2d.c")
list(APPEND TFLITE_SRCS "${TFLite220FlatBuffersSource_DIR}/src/util.cpp")

# externals - absl
file(GLOB_RECURSE ABSL_SRCS "${TFLite220AbslSource_DIR}/absl/*.cc")
file(GLOB_RECURSE ABSL_EXCLS "${TFLite220AbslSource_DIR}/absl/*test*.cc"
     "${TFLite220AbslSource_DIR}/absl/*benchmark*.cc"
     "${TFLite220AbslSource_DIR}/absl/synchronization/*.cc"
     "${TFLite220AbslSource_DIR}/absl/debugging/*.cc"
     "${TFLite220AbslSource_DIR}/absl/hash/*.cc"
     "${TFLite220AbslSource_DIR}/absl/flags/*.cc")
list(REMOVE_ITEM ABSL_SRCS ${ABSL_EXCLS})
list(APPEND TFLITE_SRCS ${ABSL_SRCS})

# Build with mmap? true
# caution: v2.2.0's Makefile has wrong code on this part. This is fixed on master branch.
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
list(APPEND TFLITE_INCLUDES "${TFLite220TensorFlowSource_DIR}")
list(APPEND TFLITE_INCLUDES "${TFLite220EigenSource_DIR}")
list(APPEND TFLITE_INCLUDES "${TFLite220AbslSource_DIR}")
list(APPEND TFLITE_INCLUDES "${TFLite220GEMMLowpSource_DIR}")
list(APPEND TFLITE_INCLUDES "${TFLite220NEON2SSESource_DIR}")
list(APPEND TFLITE_INCLUDES "${TFLite220FarmhashSource_DIR}/src")
list(APPEND TFLITE_INCLUDES "${TFLite220FlatBuffersSource_DIR}/include")
list(APPEND TFLITE_INCLUDES "${TFLite220FP16Source_DIR}/include")

add_library(tensorflow-lite-2.2.0 STATIC ${TFLITE_SRCS})
target_include_directories(tensorflow-lite-2.2.0 SYSTEM PUBLIC ${TFLITE_INCLUDES})
target_compile_definitions(tensorflow-lite-2.2.0 PUBLIC "GEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK -DTFLITE_WITH_RUY -DTFLITE_WITH_RUY_GEMV")
set_property(TARGET tensorflow-lite-2.2.0 PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(tensorflow-lite-2.2.0 eigen ${LIB_PTHREAD} dl)
if(NOT ANDROID AND ${BUILD_WITH_NNAPI})
  target_link_libraries(tensorflow-lite-2.2.0 rt)
endif()

if(ANDROID)
  target_link_libraries(tensorflow-lite-2.2.0 log)
  target_include_directories(tensorflow-lite-2.2.0 PUBLIC "${NDK_DIR}/..")
endif()
