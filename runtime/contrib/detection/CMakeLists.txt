if(NOT BUILD_DETECTION_APP)
  return()
endif(NOT BUILD_DETECTION_APP)

nnfw_find_package(Tensorflow REQUIRED)

list(APPEND SOURCES detection.cpp)

add_executable(detection ${SOURCES})
target_link_libraries(detection nnfw_lib_benchmark)
target_link_libraries(detection tensorflow-core)
