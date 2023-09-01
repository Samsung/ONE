nnas_find_package(FlatBuffers EXACT 2.0 QUIET)

if(NOT FlatBuffers_FOUND)
  message(STATUS "Build mio-tflite2121: FAILED (missing Flatbuffers 2.0)")
  return()
endif(NOT FlatBuffers_FOUND)

nnas_find_package(TensorFlowSource EXACT 2.12.1 QUIET)

if(NOT TensorFlowSource_FOUND)
  message(STATUS "Build mio-tflite2121: FAILED (missing TensorFlowSource 2.12.1)")
  return()
endif(NOT TensorFlowSource_FOUND)

message(STATUS "Build mio-tflite2121: TRUE")
message(STATUS "Build mio-tflite2121: with ${TensorFlowSource_DIR}")

set(SCHEMA_FILE "${TensorFlowSource_DIR}/tensorflow/lite/schema/schema.fbs")

# NOTE Use copy of schema.fbs as to provide unified way for circle also
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/schema.fbs"
  COMMAND ${CMAKE_COMMAND} -E copy "${SCHEMA_FILE}" schema.fbs
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS "${SCHEMA_FILE}"
)

FlatBuffers_Target(mio_tflite2121
  OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen/mio/tflite"
  INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen"
  SCHEMA_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  SCHEMA_FILES "schema.fbs"
)

add_executable(mio_tflite2121_example example.cpp)
target_link_libraries(mio_tflite2121_example mio_tflite2121)

# Temporay tflite validation tool to replace nnkit-tflite
# TODO provide full tflite validation with runtime/interpreter
add_executable(mio_tflite2121_validate example.cpp)
target_link_libraries(mio_tflite2121_validate mio_tflite2121)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(mio_tflite2121_helper STATIC ${SOURCES})
target_include_directories(mio_tflite2121_helper PRIVATE src)
target_include_directories(mio_tflite2121_helper PUBLIC include)
target_link_libraries(mio_tflite2121_helper mio_tflite2121)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(mio_tflite2121_helper_test ${TESTS})
target_include_directories(mio_tflite2121_helper_test PRIVATE src)
target_link_libraries(mio_tflite2121_helper_test mio_tflite2121)
target_link_libraries(mio_tflite2121_helper_test mio_tflite2121_helper)
