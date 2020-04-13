Protobuf_Generate(TFLCHEF_PROTO "${CMAKE_CURRENT_BINARY_DIR}/generated" "${CMAKE_CURRENT_SOURCE_DIR}" "tflchef.proto")

add_library(tflchef_proto STATIC ${TFLCHEF_PROTO_SOURCES})
target_include_directories(tflchef_proto PUBLIC ${TFLCHEF_PROTO_INCLUDE_DIRS})
target_link_libraries(tflchef_proto libprotobuf)
