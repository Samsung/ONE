set(LIB_ONERT_BACKEND_GPU_CL onert_backend_gpu_cl)

nnas_find_package(Opencl_Headers QUIET)
if(NOT Opencl_Headers_FOUND)
  return()
endif(NOT Opencl_Headers_FOUND)

if(NOT BUILD_GPU_CL)
  return()
endif(NOT BUILD_GPU_CL)

nnas_find_package(Farmhash QUIET)
if(NOT Farmhash_FOUND)
  return()
endif(NOT Farmhash_FOUND)

nnas_find_package(Abseil QUIET)
if(NOT Abseil_FOUND)
  return()
endif(NOT Abseil_FOUND)

nnas_find_package(Fp16 QUIET)
if(NOT Fp16_FOUND)
  return()
endif(NOT Fp16_FOUND)

nnfw_find_package(TensorFlowGpu REQUIRED)
if(NOT TensorFlowGpu_FOUND)
  message(FATAL_ERROR 'TensorFlowGpu lib not found')
  return()
endif(NOT TensorFlowGpu_FOUND)


file(GLOB_RECURSE SOURCES "*.cc")

add_library(${LIB_ONERT_BACKEND_GPU_CL} SHARED ${SOURCES})


target_include_directories(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
#target_include_directories(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE "${NNAS_PROJECT_SOURCE_DIR}/externals/TENSORFLOW-2.4.1")
target_include_directories(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE ${TensorFlowGpuSource_DIR})

# gpu_delegates : temporal install

#install(TARGETS tflite_gpudelegates_gpucl DESTINATION lib)

##############################################################################

target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE abseil)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE dl)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE farmhash)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE fp16)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE Headers)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE nnfw_coverage)

## gpu_delegates : link lib

target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE TensorFlowGpu)

set_target_properties(${LIB_ONERT_BACKEND_GPU_CL} PROPERTIES OUTPUT_NAME backend_gpu_cl)

if(CMAKE_BUILD_TYPE_LC STREQUAL "release")
  add_custom_command(TARGET ${LIB_ONERT_BACKEND_GPU_CL} POST_BUILD
                     COMMAND ${CMAKE_STRIP} "--strip-unneeded" $<TARGET_FILE_NAME:${LIB_ONERT_BACKEND_GPU_CL}>)
endif()

install(TARGETS ${LIB_ONERT_BACKEND_GPU_CL} DESTINATION lib)
