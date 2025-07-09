if(NOT BUILD_TFLITE_COMPARATOR_TEST_TOOL)
  message("skipping tflite comparator tool build")
  return()
endif(NOT BUILD_TFLITE_COMPARATOR_TEST_TOOL)

list(APPEND SOURCES "src/tflite_comparator.cc")
list(APPEND SOURCES "src/args.cc")

add_executable(tflite_comparator ${SOURCES})

target_link_libraries(tflite_comparator nnfw_common)
target_link_libraries(tflite_comparator nnfw-dev)
target_link_libraries(tflite_comparator nnfw_lib_tflite nnfw_lib_misc)
target_link_libraries(tflite_comparator arser)

install(TARGETS tflite_comparator DESTINATION ${CMAKE_INSTALL_BINDIR})
