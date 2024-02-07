set(LIB_ONERT_BACKEND_GPU_CL onert_backend_gpu_cl)

if(NOT BUILD_GPU_CL)
  return()
endif(NOT BUILD_GPU_CL)

nnas_find_package(Opencl_Headers QUIET)
if(NOT Opencl_Headers_FOUND)
  return()
endif(NOT Opencl_Headers_FOUND)

nnas_find_package(Farmhash QUIET)
if(NOT Farmhash_FOUND)
  return()
endif(NOT Farmhash_FOUND)

nnas_find_package(Abseil QUIET)
if(NOT Abseil_FOUND)
  return()
endif(NOT Abseil_FOUND)

nnfw_find_package(Fp16 QUIET)
if(NOT Fp16_FOUND)
  return()
endif(NOT Fp16_FOUND)

nnas_find_package(VulkanSource QUIET)
if(NOT VulkanSource_FOUND)
  return()
endif(NOT VulkanSource_FOUND)

nnas_find_package(Opengl_HeadersSource QUIET)
if(NOT Opengl_HeadersSource_FOUND)
  return()
endif(NOT Opengl_HeadersSource_FOUND)

nnas_find_package(Egl_HeadersSource QUIET)
if(NOT Egl_HeadersSource_FOUND)
  return()
endif(NOT Egl_HeadersSource_FOUND)

if (NOT ${TARGET_OS} MATCHES "tizen")
  nnas_find_package(FlatBuffers REQUIRED)
endif ()

nnfw_find_package(TensorFlowGpu QUIET)
if(NOT TensorFlowGpu_FOUND)
  message(FATAL_ERROR 'TensorFlowGpu lib not found')
  return()
endif(NOT TensorFlowGpu_FOUND)

file(GLOB_RECURSE SOURCES "*.cc")

add_library(${LIB_ONERT_BACKEND_GPU_CL} SHARED ${SOURCES})

target_include_directories(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE ${TensorFlowSource_DIR})
target_include_directories(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE ${VulkanSource_DIR}/include)
target_include_directories(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE ${Opengl_HeadersSource_DIR}/api)
target_include_directories(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE ${Egl_HeadersSource_DIR}/api)

if (${TARGET_OS} MATCHES "tizen")
    target_compile_options(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE "-Wno-error=deprecated-copy")
endif ()

target_compile_options(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE "-DCL_TARGET_OPENCL_VERSION=220" "-DEGL_NO_X11")

target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE abseil)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE dl)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE farmhash)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE OpenCL_Headers)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE fp16)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE tflite-gpu-delegate)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE ${LIB_ONERT_BACKEND_CL_COMMON})
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE nnfw_coverage)
if (${TARGET_OS} MATCHES "tizen")
  target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE flatbuffers)
else()
  target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE flatbuffers::flatbuffers)
endif ()

set_target_properties(${LIB_ONERT_BACKEND_GPU_CL} PROPERTIES OUTPUT_NAME backend_gpu_cl)

if(CMAKE_BUILD_TYPE_LC STREQUAL "release")
  add_custom_command(TARGET ${LIB_ONERT_BACKEND_GPU_CL} POST_BUILD
                     COMMAND ${CMAKE_STRIP} "--strip-unneeded" $<TARGET_FILE_NAME:${LIB_ONERT_BACKEND_GPU_CL}>)
endif()

add_library(tflite_ignore_warnings INTERFACE)
target_compile_options(tflite_ignore_warnings INTERFACE -Wno-unused-parameter -Wno-sign-compare)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
  target_compile_options(tflite_ignore_warnings INTERFACE -Wno-deprecated-copy)
endif()
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE tflite_ignore_warnings)

install(TARGETS ${LIB_ONERT_BACKEND_GPU_CL} DESTINATION lib)
