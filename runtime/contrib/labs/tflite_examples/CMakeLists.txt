add_executable(tflite_conv_example "src/conv.cpp")
target_link_libraries(tflite_conv_example tensorflow-lite ${LIB_PTHREAD} dl nnfw_lib_tflite)
