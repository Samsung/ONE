nnas_find_package(FlatBuffers EXACT 1.12 QUIET)

if(NOT FlatBuffers_FOUND)
  message(STATUS "Build mio-tflite270: FAILED (missing Flatbuffers 1.12)")
  return()
endif(NOT FlatBuffers_FOUND)

nnas_find_package(TensorFlowSource EXACT 2.7.0 QUIET)

if(NOT TensorFlowSource_FOUND)
  message(STATUS "Build mio-tflite270: FAILED (missing TensorFlowSource 2.7.0)")
  return()
endif(NOT TensorFlowSource_FOUND)

message(STATUS "Build mio-tflite270: TRUE")

set(SCHEMA_FILE "${TensorFlowSource_DIR}/tensorflow/lite/schema/schema.fbs")

# NOTE Use copy of schema.fbs as to provide unified way for circle also
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/schema.fbs"
  COMMAND ${CMAKE_COMMAND} -E copy "${SCHEMA_FILE}" schema.fbs
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS "${SCHEMA_FILE}"
)

FlatBuffers_Target(mio_tflite270
  OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen/mio/tflite"
  INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen"
  SCHEMA_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  SCHEMA_FILES "schema.fbs"
)

add_executable(mio_tflite270_example example.cpp)
target_link_libraries(mio_tflite270_example mio_tflite270)

# Temporay tflite validation tool to replace nnkit-tflite
# TODO provide full tflite validation with runtime/interpreter
add_executable(mio_tflite270_validate example.cpp)
target_link_libraries(mio_tflite270_validate mio_tflite270)

# gemmlowp is same as 2.6.0
nnas_find_package(TensorFlowGEMMLowpSource EXACT 2.6.0 QUIET)

if(NOT TensorFlowGEMMLowpSource_FOUND)
  return()
endif(NOT TensorFlowGEMMLowpSource_FOUND)

add_library(mio_tflite270_inc INTERFACE)
target_include_directories(mio_tflite270_inc SYSTEM INTERFACE "${TensorFlowSource_DIR}")
target_include_directories(mio_tflite270_inc SYSTEM INTERFACE "${TensorFlowGEMMLowpSource_DIR}")
