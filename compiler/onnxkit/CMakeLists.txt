nnas_find_package(Protobuf QUIET)
nnas_find_package(ONNXSource EXACT 1.6.0 QUIET)

if(NOT Protobuf_FOUND)
  return()
endif(NOT Protobuf_FOUND)

if(NOT ONNXSource_FOUND)
  return()
endif(NOT ONNXSource_FOUND)

message(STATUS "Build onnxkit: TRUE")

Protobuf_Generate(ONNX_PROTO
                  "${CMAKE_CURRENT_BINARY_DIR}/generated"
                  "${ONNXSource_DIR}"
                  "onnx/onnx.proto")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(onnxkitproto STATIC ${ONNX_PROTO_SOURCES})
set_target_properties(onnxkitproto PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(onnxkitproto PUBLIC ${ONNX_PROTO_INCLUDE_DIRS})
target_link_libraries(onnxkitproto PUBLIC libprotobuf)

add_executable(onnxkit ${SOURCES})
target_link_libraries(onnxkit PRIVATE stdex)
target_link_libraries(onnxkit PRIVATE cli)
target_link_libraries(onnxkit PRIVATE onnxkitproto)
target_link_libraries(onnxkit PRIVATE nncc_common)
