nnas_find_package(Protobuf QUIET)
# TensorFlowSource package is used to use ~.proto files
nnas_find_package(TensorFlowSource EXACT 1.13.1 QUIET)

if(NOT Protobuf_FOUND)
  return()
endif(NOT Protobuf_FOUND)

if(NOT TensorFlowSource_FOUND)
  return()
endif(NOT TensorFlowSource_FOUND)

message(STATUS "Build mio-tf: TRUE")

# Minimal Protocol Buffer specification for GraphDef file (.pb) encoding/decoding
unset(PROTO_FILES)
list(APPEND PROTO_FILES tensorflow/core/framework/versions.proto)
list(APPEND PROTO_FILES tensorflow/core/framework/resource_handle.proto)
list(APPEND PROTO_FILES tensorflow/core/framework/types.proto)
list(APPEND PROTO_FILES tensorflow/core/framework/tensor.proto)
list(APPEND PROTO_FILES tensorflow/core/framework/tensor_shape.proto)
list(APPEND PROTO_FILES tensorflow/core/framework/attr_value.proto)
list(APPEND PROTO_FILES tensorflow/core/framework/op_def.proto)
list(APPEND PROTO_FILES tensorflow/core/framework/node_def.proto)
list(APPEND PROTO_FILES tensorflow/core/framework/function.proto)
list(APPEND PROTO_FILES tensorflow/core/framework/graph.proto)

Protobuf_Generate(GRAPHDEF_PROTO
                  "${CMAKE_CURRENT_BINARY_DIR}/generated"
                  "${TensorFlowSource_DIR}"
                  ${PROTO_FILES})

add_library(mio_tf STATIC ${GRAPHDEF_PROTO_SOURCES})
set_target_properties(mio_tf PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mio_tf PUBLIC ${GRAPHDEF_PROTO_INCLUDE_DIRS})
target_link_libraries(mio_tf PUBLIC ${GRAPHDEF_PROTO_LIBRARIES})

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

file(GLOB_RECURSE TESTS "src/*.test.cpp")

GTest_AddTest(mio_tf_test ${TESTS})
target_include_directories(mio_tf_test PRIVATE src)
target_link_libraries(mio_tf_test mio_tf)
