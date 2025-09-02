nnas_find_package(FlatBuffers EXACT 23.5.26 QUIET)

if(NOT FlatBuffers_FOUND)
  message(STATUS "Build mio-tflite: FAILED (missing Flatbuffers 23.5.26)")
  return()
endif(NOT FlatBuffers_FOUND)

nnas_find_package(TensorFlowSource EXACT 2.19.0 QUIET)

if(NOT TensorFlowSource_FOUND)
  message(STATUS "Build mio-tflite: FAILED (missing TensorFlowSource 2.19.0)")
  return()
endif(NOT TensorFlowSource_FOUND)

message(STATUS "Build mio-tflite: TRUE")
message(STATUS "Build mio-tflite: with ${TensorFlowSource_DIR}")

set(SCHEMA_FILE "${TensorFlowSource_DIR}/tensorflow/compiler/mlir/lite/schema/schema.fbs")

# NOTE Use copy of schema.fbs as to provide unified way for circle also
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/schema.fbs"
  COMMAND ${CMAKE_COMMAND} -E copy "${SCHEMA_FILE}" schema.fbs
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS "${SCHEMA_FILE}"
)

FlatBuffersMuteable_Target(mio_tflite
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

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(mio_tflite_helper STATIC ${SOURCES})
target_include_directories(mio_tflite_helper PRIVATE src)
target_include_directories(mio_tflite_helper PUBLIC include)
target_link_libraries(mio_tflite_helper mio_tflite)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(mio_tflite_helper_test ${TESTS})
target_include_directories(mio_tflite_helper_test PRIVATE src)
target_link_libraries(mio_tflite_helper_test mio_tflite)
target_link_libraries(mio_tflite_helper_test mio_tflite_helper)
