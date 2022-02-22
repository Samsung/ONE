nnas_find_package(FlatBuffers EXACT 2.0 QUIET)

if(NOT FlatBuffers_FOUND)
  message(STATUS "mio-circle skip: FlatBuffers 1.10 NOT FOUND")
  return()
endif(NOT FlatBuffers_FOUND)

message(STATUS "Build mio-circle: TRUE")

# TODO Find a better way
set(SCHEMA_FILE "${NNAS_PROJECT_SOURCE_DIR}/nnpackage/schema/circle_schema.fbs")

# NOTE Copy circle_schema.fbs as schema.fbs to generate "schema_generated.fbs" instead of "circle_schema_generated.fbs"
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/schema.fbs"
  COMMAND ${CMAKE_COMMAND} -E copy "${SCHEMA_FILE}" schema.fbs
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS "${SCHEMA_FILE}"
)

FlatBuffers_Target(mio_circle
  OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen/mio/circle"
  INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen"
  SCHEMA_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  SCHEMA_FILES "schema.fbs"
)

# This example shows how to use "mio-circle" library
add_executable(mio_circle_example example.cpp)
target_link_libraries(mio_circle_example mio_circle)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(mio_circle_helper STATIC ${SOURCES})
target_include_directories(mio_circle_helper PRIVATE src)
target_include_directories(mio_circle_helper PUBLIC include)
target_link_libraries(mio_circle_helper mio_circle)
