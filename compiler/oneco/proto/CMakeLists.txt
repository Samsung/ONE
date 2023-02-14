# Minimal Protocol Buffer specification for ModelProto file (.pb) encoding/decoding
unset(PROTO_FILES)
list(APPEND PROTO_FILES onnx/onnx.proto)

Protobuf_Generate(ONNX_MODEL_PROTO "${CMAKE_CURRENT_BINARY_DIR}/generated" "${ONNXSource_DIR}" ${PROTO_FILES})

add_library(moco_onnx_proto STATIC ${ONNX_MODEL_PROTO_SOURCES})
set_target_properties(moco_onnx_proto PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(moco_onnx_proto PUBLIC ${ONNX_MODEL_PROTO_INCLUDE_DIRS})
target_link_libraries(moco_onnx_proto PUBLIC libprotobuf)
