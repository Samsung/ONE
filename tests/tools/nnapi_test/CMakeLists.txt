list(APPEND SOURCES "src/nnapi_test.cc")

add_executable(nnapi_test ${SOURCES})
target_link_libraries(nnapi_test nnfw_lib_tflite)
install(TARGETS nnapi_test DESTINATION bin)
