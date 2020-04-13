if(NOT BUILD_TFLITE_LOADER_TEST_TOOL)
  message("skipping tflite loader tool build")
  return()
endif(NOT BUILD_TFLITE_LOADER_TEST_TOOL)

if(NOT BUILD_NEURUN)
  message("skipping tflite loader tool build: neurun is not built")
  return()
endif(NOT BUILD_NEURUN)

list(APPEND SOURCES "src/tflite_loader.cc")
list(APPEND SOURCES "src/args.cc")

nnas_find_package(Boost REQUIRED)

add_executable(tflite_loader_test_tool ${SOURCES})
target_include_directories(tflite_loader_test_tool PRIVATE ${Boost_INCLUDE_DIRS})

target_link_libraries(tflite_loader_test_tool neurun_core neurun tflite_loader)
target_link_libraries(tflite_loader_test_tool nnfw_lib_tflite tensorflow-lite ${LIB_PTHREAD} dl nnfw_lib_misc)
target_link_libraries(tflite_loader_test_tool boost_program_options boost_system boost_filesystem)

install(TARGETS tflite_loader_test_tool DESTINATION bin)
