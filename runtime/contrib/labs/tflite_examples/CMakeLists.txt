add_executable(tflite_conv_example "src/conv.cpp")
target_link_libraries(tflite_conv_example tensorflow-lite Threads::Threads dl nnfw_lib_tflite)
