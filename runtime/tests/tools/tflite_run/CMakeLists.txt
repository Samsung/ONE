if(NOT BUILD_TFLITE_RUN)
  return()
endif(NOT BUILD_TFLITE_RUN)

list(APPEND TFLITE_RUN_SRCS "src/tflite_run.cc")
list(APPEND TFLITE_RUN_SRCS "src/args.cc")
list(APPEND TFLITE_RUN_SRCS "src/tensor_dumper.cc")
list(APPEND TFLITE_RUN_SRCS "src/tensor_loader.cc")

add_executable(tflite_run ${TFLITE_RUN_SRCS})

target_link_libraries(tflite_run nnfw_common)
target_link_libraries(tflite_run nnfw_lib_tflite nnfw_lib_benchmark)
target_link_libraries(tflite_run arser)

install(TARGETS tflite_run DESTINATION ${CMAKE_INSTALL_BINDIR})

## Add test cpp file
add_executable(tflite_test src/tflite_test.cc)
target_link_libraries(tflite_test nnfw_common)
## Link test executable against gtest & gtest_main
target_link_libraries(tflite_test gtest gtest_main Threads::Threads)
## install test binary for packaging
install(TARGETS tflite_test DESTINATION unittest)
