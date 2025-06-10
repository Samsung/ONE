# check gtest is OK
add_subdirectory(check-gtest)
# generate onnx models
add_subdirectory(gen-onnx)
# model conversion with onnx2circle
add_subdirectory(onnx2circle-models)
# value test of source onnx vs circle
add_subdirectory(onnx2circle-value-test)
# rewrite test of circle
add_subdirectory(onnx2circle-rewrite-test)
# circle import/export test
add_subdirectory(circle-impexp-test)
