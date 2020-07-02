message(STATUS "Build TensorFlowProtoText from '${TensorFlowSource_DIR}'")

#
# Build "proto_text" tool
#
unset(SOURCE_FILES)

macro(Source_Add RPATH)
  list(APPEND SOURCE_FILES "${TensorFlowSource_DIR}/${RPATH}")
endmacro(Source_Add)

# This list comes from "tensorflow/contrib/makefile/proto_text_cc_files.txt"
Source_Add(tensorflow/core/lib/core/status.cc)
Source_Add(tensorflow/core/lib/core/threadpool.cc)
Source_Add(tensorflow/core/lib/hash/hash.cc)
Source_Add(tensorflow/core/lib/io/inputstream_interface.cc)
Source_Add(tensorflow/core/lib/io/random_inputstream.cc)
Source_Add(tensorflow/core/lib/io/buffered_inputstream.cc)
Source_Add(tensorflow/core/lib/io/inputbuffer.cc)
Source_Add(tensorflow/core/lib/io/iterator.cc)
Source_Add(tensorflow/core/lib/io/path.cc)
Source_Add(tensorflow/core/lib/strings/numbers.cc)
Source_Add(tensorflow/core/lib/strings/scanner.cc)
Source_Add(tensorflow/core/lib/strings/str_util.cc)
Source_Add(tensorflow/core/lib/strings/strcat.cc)
Source_Add(tensorflow/core/lib/strings/stringprintf.cc)
Source_Add(tensorflow/core/lib/strings/proto_text_util.cc)
Source_Add(tensorflow/core/platform/cpu_info.cc)
Source_Add(tensorflow/core/platform/denormal.cc)
Source_Add(tensorflow/core/platform/env.cc)
Source_Add(tensorflow/core/platform/env_time.cc)
Source_Add(tensorflow/core/platform/file_system.cc)
Source_Add(tensorflow/core/platform/file_system_helper.cc)
Source_Add(tensorflow/core/platform/protobuf_util.cc)
Source_Add(tensorflow/core/platform/setround.cc)
Source_Add(tensorflow/core/platform/tracing.cc)
Source_Add(tensorflow/core/platform/posix/env.cc)
Source_Add(tensorflow/core/platform/posix/env_time.cc)
Source_Add(tensorflow/core/platform/posix/error.cc)
Source_Add(tensorflow/core/platform/posix/load_library.cc)
Source_Add(tensorflow/core/platform/posix/port.cc)
Source_Add(tensorflow/core/platform/posix/posix_file_system.cc)
Source_Add(tensorflow/core/platform/default/logging.cc)
Source_Add(tensorflow/core/platform/default/mutex.cc)
Source_Add(tensorflow/core/platform/default/protobuf.cc)

Source_Add(tensorflow/tools/proto_text/gen_proto_text_functions_lib.cc)
Source_Add(tensorflow/tools/proto_text/gen_proto_text_functions.cc)

unset(PROTO_FILES)

macro(Proto_Add RPATH)
  list(APPEND PROTO_FILES "${RPATH}")
endmacro(Proto_Add)

Proto_Add(tensorflow/core/lib/core/error_codes.proto)
Proto_Add(tensorflow/core/framework/types.proto)
Proto_Add(tensorflow/core/framework/tensor.proto)
Proto_Add(tensorflow/core/framework/tensor_shape.proto)
Proto_Add(tensorflow/core/framework/summary.proto)
Proto_Add(tensorflow/core/framework/resource_handle.proto)

Protobuf_Generate(PROTO_TEXT_PROTO
  "${CMAKE_CURRENT_BINARY_DIR}/generated/proto_text"
  "${TensorFlowSource_DIR}"
  ${PROTO_FILES}
)

add_executable(tensorflow-prototext-1.12 ${SOURCE_FILES} ${PROTO_TEXT_PROTO_SOURCES})
target_include_directories(tensorflow-prototext-1.12 PRIVATE ${TensorFlowSource_DIR})
target_include_directories(tensorflow-prototext-1.12 PRIVATE ${PROTO_TEXT_PROTO_INCLUDE_DIRS})

target_link_libraries(tensorflow-prototext-1.12 PRIVATE abseil)
target_link_libraries(tensorflow-prototext-1.12 PRIVATE eigen-fd6845384b86)
target_link_libraries(tensorflow-prototext-1.12 PRIVATE ${PROTO_TEXT_PROTO_LIBRARIES})
target_link_libraries(tensorflow-prototext-1.12 PRIVATE Google::DoubleConversion)
target_link_libraries(tensorflow-prototext-1.12 PRIVATE Google::NSync)
target_link_libraries(tensorflow-prototext-1.12 PRIVATE dl)
