#
# Tensorflow Lite GPU delegate library 2.8.0
#

set(LIB_TENSORFLOW_GPU_DELEGATE tensorflow-lite-gpu)

#TENSORFLOWGPU_SOURCE_DIR
set(TENSORFLOWSOURCE_DIR ${TensorFlowSource_DIR})
set(TENSORFLOW_LITE_BASE ${TENSORFLOWSOURCE_DIR}/tensorflow/lite)
set(REF_TENSORFLOW_LITE_GPU_DELEGATE_SRC_BASE "${TENSORFLOW_LITE_BASE}/delegates/gpu")

set(SRC_BASE "${REF_TENSORFLOW_LITE_GPU_DELEGATE_SRC_BASE}")
file(GLOB GPU_CL_SRC_LIST   "${SRC_BASE}/cl/*.cc"
                                    "${SRC_BASE}/cl/kernels/*.cc"
                                    "${SRC_BASE}/common/*.cc"
                                    "${SRC_BASE}/common/selectors/*.cc"
                                    "${SRC_BASE}/common/selectors/default/*.cc"
                                    "${SRC_BASE}/common/task/*.cc"
                                    "${SRC_BASE}/common/tasks/*.cc"
                                    "${SRC_BASE}/common/tasks/special/*.cc"
                                    "${SRC_BASE}/common/memory_management/*.cc"
                                    "${SRC_BASE}/common/transformations/*.cc"
                                     )

file(GLOB REMOVE_TEST_SRCS          "${SRC_BASE}/cl/*_test*.cc"
                                    "${SRC_BASE}/cl/testing/*.cc"
                                    "${SRC_BASE}/cl/kernels/*_test*.cc"
                                    "${SRC_BASE}/common/*_test*.cc"
                                    "${SRC_BASE}/common/tasks/*_test*.cc"
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

list(APPEND GPU_CL_SRC_LIST "${TENSORFLOW_LITE_BASE}/experimental/acceleration/compatibility/android_info.cc")

list(REMOVE_ITEM GPU_CL_SRC_LIST ${REMOVE_TEST_SRCS})
list(REMOVE_ITEM GPU_CL_SRC_LIST ${REMOVE_SRCS})
list(APPEND TFLITE_GPU_SRCS ${GPU_CL_SRC_LIST})

add_library(${LIB_TENSORFLOW_GPU_DELEGATE} STATIC ${TFLITE_GPU_SRCS})

target_include_directories(${LIB_TENSORFLOW_GPU_DELEGATE} PUBLIC "${Opencl_HeadersSource_DIR}")
target_include_directories(${LIB_TENSORFLOW_GPU_DELEGATE} PUBLIC "${Fp16Source_DIR}/include")
target_include_directories(${LIB_TENSORFLOW_GPU_DELEGATE} PUBLIC "${TensorFlowSource_DIR}")
target_include_directories(${LIB_TENSORFLOW_GPU_DELEGATE} PUBLIC "${TensorFlowGEMMLowpSource_DIR}")
target_include_directories(${LIB_TENSORFLOW_GPU_DELEGATE} PUBLIC "${TensorFlowEigenSource_DIR}")
target_include_directories(${LIB_TENSORFLOW_GPU_DELEGATE} PUBLIC "${VulkanSource_DIR}/include")
target_include_directories(${LIB_TENSORFLOW_GPU_DELEGATE} PUBLIC "${Opengl_HeadersSource_DIR}/api")
target_include_directories(${LIB_TENSORFLOW_GPU_DELEGATE} PUBLIC "${Egl_HeadersSource_DIR}/api")

target_link_libraries(${LIB_TENSORFLOW_GPU_DELEGATE} PUBLIC abseil farmhash flatbuffers::flatbuffers-2.0)

# GL codes are not used on gpu_cl
target_compile_options(${LIB_TENSORFLOW_GPU_DELEGATE} PRIVATE "-DCL_DELEGATE_NO_GL")
target_compile_options(${LIB_TENSORFLOW_GPU_DELEGATE} PRIVATE "-DCL_TARGET_OPENCL_VERSION=220")
target_compile_options(${LIB_TENSORFLOW_GPU_DELEGATE} PRIVATE "-DTFLITE_GPU_BINARY_RELEASE" "-DEGL_NO_X11")

# deprecated-copy warning on header (gcc 9.4.0)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.4)
  target_compile_options(${LIB_TENSORFLOW_GPU_DELEGATE} PUBLIC "-Wno-deprecated-copy")
endif()

# Applying PIC first, currently used on gpu_cl only
set_target_properties(${LIB_TENSORFLOW_GPU_DELEGATE} PROPERTIES POSITION_INDEPENDENT_CODE ON)
