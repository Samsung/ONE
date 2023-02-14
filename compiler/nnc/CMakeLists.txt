set(NNC_ROOT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(NNC_ROOT_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH ${NNC_ROOT_SRC_DIR}/cmake)

include(config)
include(utils)

# configure file that contains extern definitions
configure_file(${NNC_ROOT_SRC_DIR}/include/Definitions.h.in ${NNC_ROOT_BIN_DIR}/include/Definitions.h)

# add interface header files

# target for compiler executable
add_executable(${NNC_TARGET_EXECUTABLE} ${NNC_DRIVER_DIR}/main.cpp ${NNC_DRIVER_DIR}/Driver.cpp
                                        ${NNC_DRIVER_DIR}/Options.cpp)

# install compiler
nnc_install_executable(${NNC_TARGET_EXECUTABLE})
if(NNC_FRONTEND_CAFFE2_ENABLED)
  target_link_libraries(${NNC_TARGET_EXECUTABLE} PRIVATE mir_caffe2_importer)
endif()
if(NNC_FRONTEND_CAFFE_ENABLED)
  target_link_libraries(${NNC_TARGET_EXECUTABLE} PRIVATE mir_caffe_importer)
endif()
if(NNC_FRONTEND_TFLITE_ENABLED)
  target_link_libraries(${NNC_TARGET_EXECUTABLE} PRIVATE mir_tflite_importer)
endif()
if(NNC_FRONTEND_ONNX_ENABLED)
  target_link_libraries(${NNC_TARGET_EXECUTABLE} PRIVATE mir_onnx_importer)
endif()

add_subdirectory(support)
add_subdirectory(pass)
add_subdirectory(passes)
add_subdirectory(backends)
if(NNC_ENABLE_UNITTEST)
  add_subdirectory(unittests)
endif()
add_subdirectory(tests)
add_subdirectory(utils)
