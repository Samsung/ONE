Protobuf_Generate(CIRCLECHEF_PROTO "${CMAKE_CURRENT_BINARY_DIR}/generated" "${CMAKE_CURRENT_SOURCE_DIR}"
                  "circlechef.proto")

add_library(circlechef_proto STATIC ${CIRCLECHEF_PROTO_SOURCES})
target_include_directories(circlechef_proto PUBLIC ${CIRCLECHEF_PROTO_INCLUDE_DIRS})
target_link_libraries(circlechef_proto libprotobuf)
