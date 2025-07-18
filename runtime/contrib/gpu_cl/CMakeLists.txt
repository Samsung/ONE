set(LIB_ONERT_BACKEND_GPU_CL onert_backend_gpu_cl)

if(NOT BUILD_GPU_CL)
  return()
endif(NOT BUILD_GPU_CL)

nnfw_find_package(TensorFlowGpu QUIET)
if(NOT TensorFlowGpu_FOUND)
  message(FATAL_ERROR 'TensorFlowGpu lib not found')
  return()
endif(NOT TensorFlowGpu_FOUND)

file(GLOB_RECURSE SOURCES "*.cc")

add_library(${LIB_ONERT_BACKEND_GPU_CL} SHARED ${SOURCES})

target_include_directories(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if (${TARGET_OS} MATCHES "tizen")
    target_compile_options(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE "-Wno-error=deprecated-copy")
endif ()

target_compile_definitions(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE CL_TARGET_OPENCL_VERSION=220 EGL_NO_X11)

target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE dl)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE tflite-gpu-delegate)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE ${LIB_ONERT_BACKEND_CL_COMMON})
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE nnfw_common)

set_target_properties(${LIB_ONERT_BACKEND_GPU_CL} PROPERTIES OUTPUT_NAME backend_gpu_cl INSTALL_RPATH ${ONERT_RPATH_PLUGIN})

add_library(tflite_ignore_warnings INTERFACE)
target_compile_options(tflite_ignore_warnings INTERFACE -Wno-unused-parameter -Wno-sign-compare)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
  target_compile_options(tflite_ignore_warnings INTERFACE -Wno-deprecated-copy)
endif()
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE tflite_ignore_warnings)

install(TARGETS ${LIB_ONERT_BACKEND_GPU_CL} DESTINATION ${ONERT_INSTALL_BACKENDDIR})
