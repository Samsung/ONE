
#
#  "tflite_model_generator" is a generator random TFLite models.
#
add_subdirectory(tflite_model_generator)

# dumpers of NN models
add_subdirectory(tflite_dot_dumper)
add_subdirectory(caffe_dot_dumper)
add_subdirectory(caffe2_dot_dumper)

# input tensors generator
add_subdirectory(input_gen)
