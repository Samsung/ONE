nnas_find_package(FlatBuffers EXACT 23.5.26 QUIET)

if(NOT FlatBuffers_FOUND)
  message(STATUS "Build mio-tflite: FAILED (missing Flatbuffers)")
  return()
endif(NOT FlatBuffers_FOUND)

nnas_find_package(TensorFlowSource EXACT 2.3.0 QUIET)

if(NOT TensorFlowSource_FOUND)
  return()
endif(NOT TensorFlowSource_FOUND)

message(STATUS "Build mio-tflite: TRUE")

set(SCHEMA_FILE "${TensorFlowSource_DIR}/tensorflow/lite/schema/schema.fbs")

# NOTE Use copy of schema.fbs as to provide unified way for circle also
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/schema.fbs"
  COMMAND ${CMAKE_COMMAND} -E copy "${SCHEMA_FILE}" schema.fbs
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS "${SCHEMA_FILE}"
)

FlatBuffers_Target(mio_tflite
  OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen/mio/tflite"
  INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen"
  SCHEMA_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  SCHEMA_FILES "schema.fbs"
)

add_executable(mio_tflite_example example.cpp)
target_link_libraries(mio_tflite_example mio_tflite)

# Temporay tflite validation tool to replace nnkit-tflite
# TODO provide full tflite validation with runtime/interpreter
add_executable(mio_tflite_validate example.cpp)
target_link_libraries(mio_tflite_validate mio_tflite)

nnas_find_package(TensorFlowGEMMLowpSource EXACT 2.3.0 QUIET)

if(NOT TensorFlowGEMMLowpSource_FOUND)
  return()
endif(NOT TensorFlowGEMMLowpSource_FOUND)

add_library(mio_tflite_inc INTERFACE)
target_include_directories(mio_tflite_inc SYSTEM INTERFACE "${TensorFlowSource_DIR}")
target_include_directories(mio_tflite_inc SYSTEM INTERFACE "${TensorFlowGEMMLowpSource_DIR}")
