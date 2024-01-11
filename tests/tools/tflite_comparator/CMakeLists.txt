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

nnfw_find_package(Boost REQUIRED program_options)

add_executable(tflite_comparator ${SOURCES})
target_include_directories(tflite_comparator PRIVATE ${Boost_INCLUDE_DIRS})

target_link_libraries(tflite_comparator nnfw-dev)
target_link_libraries(tflite_comparator nnfw_lib_tflite nnfw_lib_misc)
target_link_libraries(tflite_comparator ${Boost_PROGRAM_OPTIONS_LIBRARY})

install(TARGETS tflite_comparator DESTINATION bin)
