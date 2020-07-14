nnas_find_package(FlatBuffers QUIET)

if(NOT FlatBuffers_FOUND)
  message(STATUS "FlatBuffers package not found")
  return()
endif(NOT FlatBuffers_FOUND)

# TODO recover official release version
# NOTE we cannot use version number like "2.3.0-rc0" for find_package()
#      use TensorFlowSource-2.3.0-rc0 as config itself
# nnas_find_package(TensorFlowSource EXACT 2.3.0 QUIET)
nnas_find_package(TensorFlowSource-2.3.0-rc0 QUIET)

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
