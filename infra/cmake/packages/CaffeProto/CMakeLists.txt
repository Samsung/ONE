Protobuf_Generate(CAFFE_PROTO "${CMAKE_CURRENT_BINARY_DIR}/generated/caffe" "${CaffeSource_DIR}/src"
                  "caffe/proto/caffe.proto")

add_library(caffeproto STATIC ${CAFFE_PROTO_SOURCES})
set_target_properties(caffeproto PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(caffeproto PUBLIC ${CAFFE_PROTO_INCLUDE_DIRS})
target_link_libraries(caffeproto libprotobuf)
