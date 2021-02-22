file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(nnkit_support_onnx-1.4 STATIC ${SOURCES})
set_target_properties(nnkit_support_onnx-1.4 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nnkit_support_onnx-1.4 PUBLIC include)
target_link_libraries(nnkit_support_onnx-1.4 nnkit_intf_backend)
target_link_libraries(nnkit_support_onnx-1.4 onnxruntime)

add_library(nnkit_support_onnx ALIAS nnkit_support_onnx-1.4)
