# check gtest is OK
add_subdirectory(check-gtest)
# generate onnx models
add_subdirectory(gen-onnx)
# model conversion with onnx2circle
add_subdirectory(onnx2circle-models)
