if(NOT BUILD_TFLITE_LOADER AND NOT BUILD_CIRCLE_LOADER)
  return()
endif(NOT BUILD_TFLITE_LOADER AND NOT BUILD_CIRCLE_LOADER)

nnfw_find_package(FlatBuffersSource REQUIRED)

add_library(base_loader INTERFACE)
target_include_directories(base_loader INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(base_loader INTERFACE ${FlatBuffersSource_DIR}/include)

target_link_libraries(base_loader INTERFACE onert_core)
