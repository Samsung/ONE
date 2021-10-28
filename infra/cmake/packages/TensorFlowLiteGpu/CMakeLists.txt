#
# Tensorflow Lite GPU delegate library 2.4.1
#

set(LIB_TENSORFLOW_GPU_DELEGATE "TensorFlowGpu")

#TENSORFLOWGPU_SOURCE_DIR
set(REF_TENSORFLOW_SRC_BASE ${TENSORFLOWGPU_SOURCE_DIR})
set(REF_TENSORFLOW_LITE_SRC_BASE ${REF_TENSORFLOW_SRC_BASE}/tensorflow/lite)
set(REF_TENSORFLOW_LITE_GPU_DELEGATE_SRC_BASE "${REF_TENSORFLOW_LITE_SRC_BASE}/delegates/gpu")

set(SRC_BASE "${REF_TENSORFLOW_LITE_GPU_DELEGATE_SRC_BASE}")
file(GLOB GPU_CL_SRC_LIST   "${SRC_BASE}/cl/*.cc"
                                    "${SRC_BASE}/cl/kernels/*.cc"
                                    "${SRC_BASE}/cl/kernels/special/*.cc"
                                    "${SRC_BASE}/cl/kernels/selectors/*.cc"
                                    "${SRC_BASE}/cl/selectors/*.cc"
                                    "${SRC_BASE}/common/*.cc"
# Available, but not needed yet
#                                    "${SRC_BASE}/common/default/*.cc"
#                                    "${SRC_BASE}/common/memory_managements/*.cc"
#                                    "${SRC_BASE}/common/transformations/*.cc"
                                     )

file(GLOB GPU_CL_HDRS_GLOB   "${SRC_BASE}/cl/*.h"
                                    "${SRC_BASE}/cl/kernels/*.h"
                                    "${SRC_BASE}/cl/kernels/special/*.h"
                                    "${SRC_BASE}/cl/kernels/selectors/*.h"
                                    "${SRC_BASE}/cl/selectors/*.h"
                                    "${SRC_BASE}/common/*.h"
                                    "${SRC_BASE}/common/default/*.h"
                                    "${SRC_BASE}/common/memory_managements/*.h"
                                    "${SRC_BASE}/common/transformations/*.h"
                                    )
list(APPEND GPU_CL_SRC_LIST "${_GPU_CL_HDRS_GLOB}")

file(GLOB REMOVE_TEST_SRCS          "${SRC_BASE}/cl/*_test*.cc"
                                    "${SRC_BASE}/cl/testing/*.cc"
                                    "${SRC_BASE}/cl/kernels/*_test*.cc"
                                    "${SRC_BASE}/common/*_test*.cc"
                                    "${SRC_BASE}/common/transformations/*_test*.cc"
                                    )
# Not available
file(GLOB REMOVE_SRCS               "${SRC_BASE}/cl/*gl*.cc"
                                    "${SRC_BASE}/cl/gpu_api_delegate.cc"
                                    "${SRC_BASE}/cl/serialization.cc"
                                    "${SRC_BASE}/common/lstm_parser.cc"
                                    "${SRC_BASE}/common/model_builder.cc"
                                    "${SRC_BASE}/common/model_builder_helper.cc"
                                    "${SRC_BASE}/common/object_reader.cc"
                                    "${SRC_BASE}/common/quantization_util.cc"
                                    "${SRC_BASE}/common/memory_management/*_test.cc"
                                    )

list(REMOVE_ITEM GPU_CL_SRC_LIST ${REMOVE_TEST_SRCS})
list(REMOVE_ITEM GPU_CL_SRC_LIST ${REMOVE_SRCS})
list(APPEND TFLITE_GPU_SRCS ${GPU_CL_SRC_LIST})

add_library(${LIB_TENSORFLOW_GPU_DELEGATE} STATIC ${TFLITE_GPU_SRCS})
target_include_directories(${LIB_TENSORFLOW_GPU_DELEGATE} PRIVATE "${Fp16Source_DIR}/include")
target_include_directories(${LIB_TENSORFLOW_GPU_DELEGATE} PRIVATE "${TENSORFLOWGPU_SOURCE_DIR}")
target_link_libraries(${LIB_TENSORFLOW_GPU_DELEGATE} PRIVATE abseil farmhash fp16)

add_library(tflitegpu_ignore_warnings INTERFACE)
target_compile_options(tflitegpu_ignore_warnings INTERFACE -Wno-unused-parameter -Wno-sign-compare)
target_link_libraries(${LIB_TENSORFLOW_GPU_DELEGATE} INTERFACE tflitegpu_ignore_warnings)

# GL codes are not used on gpu_cl
target_compile_options(${LIB_TENSORFLOW_GPU_DELEGATE} PRIVATE "-DCL_DELEGATE_NO_GL")

# Applying PIC first, currently used on gpu_cl only
set_target_properties(${LIB_TENSORFLOW_GPU_DELEGATE} PROPERTIES POSITION_INDEPENDENT_CODE ON)
