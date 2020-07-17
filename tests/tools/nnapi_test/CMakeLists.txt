if(NOT BUILD_NNAPI_TEST)
  return()
endif(NOT BUILD_NNAPI_TEST)

list(APPEND SOURCES "src/nnapi_test.cc")
list(APPEND SOURCES "src/args.cc")

nnfw_find_package(Boost REQUIRED program_options)

add_executable(nnapi_test ${SOURCES})
target_include_directories(nnapi_test PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(nnapi_test nnfw_lib_tflite)
target_link_libraries(nnapi_test ${Boost_PROGRAM_OPTIONS_LIBRARY})
install(TARGETS nnapi_test DESTINATION bin)
