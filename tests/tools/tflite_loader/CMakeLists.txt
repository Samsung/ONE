if(NOT BUILD_TFLITE_LOADER_TEST_TOOL)
  message("skipping tflite loader tool build")
  return()
endif(NOT BUILD_TFLITE_LOADER_TEST_TOOL)

if(NOT BUILD_ONERT)
  message("skipping tflite loader tool build: onert is not built")
  return()
endif(NOT BUILD_ONERT)

list(APPEND SOURCES "src/tflite_loader.cc")
list(APPEND SOURCES "src/args.cc")

nnfw_find_package(Boost REQUIRED program_options system filesystem)

add_executable(tflite_loader_test_tool ${SOURCES})
target_include_directories(tflite_loader_test_tool PRIVATE ${Boost_INCLUDE_DIRS})

target_link_libraries(tflite_loader_test_tool nnfw-dev)
target_link_libraries(tflite_loader_test_tool nnfw_lib_tflite nnfw_lib_misc)
target_link_libraries(tflite_loader_test_tool ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY})

install(TARGETS tflite_loader_test_tool DESTINATION bin)
