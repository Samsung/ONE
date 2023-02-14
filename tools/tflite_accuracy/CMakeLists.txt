if(NOT BUILD_TFLITE_ACCURACY)
  return()
endif(NOT BUILD_TFLITE_ACCURACY)

list(APPEND TFLITE_ACCURACY_SRCS "src/tflite_accuracy.cc")

add_executable(tflite_accuracy ${TFLITE_ACCURACY_SRCS})
target_include_directories(tflite_accuracy PRIVATE src)
target_link_libraries(tflite_accuracy tensorflow-lite ${LIB_PTHREAD} dl nnfw_lib_tflite)
target_link_libraries(tflite_accuracy boost_program_options boost_system boost_filesystem)

install(TARGETS tflite_accuracy DESTINATION bin)
