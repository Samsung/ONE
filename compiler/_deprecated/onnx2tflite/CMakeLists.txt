nnas_include(TargetRequire)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(onnx2tflite ${SOURCES})
target_link_libraries(onnx2tflite mir_onnx_importer)
target_link_libraries(onnx2tflite mir2loco)
target_link_libraries(onnx2tflite exo)
