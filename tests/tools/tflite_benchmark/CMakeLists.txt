if (NOT BUILD_TFLITE_BENCHMARK)
  return()
endif(NOT BUILD_TFLITE_BENCHMARK)

list(APPEND SOURCES "src/tflite_benchmark.cc")

add_executable(tflite_benchmark ${SOURCES})
target_link_libraries(tflite_benchmark nnfw_lib_tflite tensorflow-lite ${LIB_PTHREAD} dl nnfw_lib_misc)

install(TARGETS tflite_benchmark DESTINATION bin)
