if(NOT BUILD_TFLITE_COMPARATOR_TEST_TOOL)
  message("skipping tflite comparator tool build")
  return()
endif(NOT BUILD_TFLITE_COMPARATOR_TEST_TOOL)

if(NOT BUILD_ONERT)
  message("skipping tflite comparator tool build: onert is not built")
  return()
endif(NOT BUILD_ONERT)

list(APPEND SOURCES "src/tflite_comparator.cc")
list(APPEND SOURCES "src/args.cc")

add_executable(tflite_comparator ${SOURCES})

target_link_libraries(tflite_comparator nnfw-dev)
target_link_libraries(tflite_comparator nnfw_lib_tflite nnfw_lib_misc)
target_link_libraries(tflite_comparator cli11)

install(TARGETS tflite_comparator DESTINATION bin)
