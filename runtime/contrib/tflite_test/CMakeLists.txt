if(NOT BUILD_TFLITE_TEST)
  return()
endif(NOT BUILD_TFLITE_TEST)

nnfw_find_package(Tensorflow QUIET)

if(NOT Tensorflow_FOUND)
  return()
endif(NOT Tensorflow_FOUND)

list(APPEND SOURCES tflite_test.cpp)

add_executable(tflite_test ${SOURCES})
target_link_libraries(tflite_test PRIVATE nnfw_lib_tflite)
target_link_libraries(tflite_test PRIVATE tensorflow-lite)
target_link_libraries(tflite_test PRIVATE tensorflow-core)
