nnas_find_package(FlatBuffers EXACT 23.5.26 QUIET)
if(NOT FlatBuffers_FOUND)
  message(STATUS "Configure pics: FAILED (missing FlatBuffers)")
  return()
endif(NOT FlatBuffers_FOUND)

unset(PICS_DEPS)

###
### Generate python interface for circle schema
###
set(CIRCLE_SCHEMA_PYTHON_DIR "${CMAKE_CURRENT_BINARY_DIR}/circle")

get_target_property(SCHEMA_BIN_PATH mio_circle09 BINARY_DIR)

add_custom_command(
  OUTPUT ${CIRCLE_SCHEMA_PYTHON_DIR}
  COMMAND "$<TARGET_FILE:flatbuffers::flatc>" --python
          -o "${CMAKE_CURRENT_BINARY_DIR}" "${SCHEMA_BIN_PATH}/schema.fbs"
  DEPENDS flatbuffers::flatc
  COMMENT "Generate python interface for circle schema"
)

list(APPEND PICS_DEPS "${CIRCLE_SCHEMA_PYTHON_DIR}")

# This enforces CMake to generate all the dependencies during "build" phase
add_custom_target(pics ALL DEPENDS ${PICS_DEPS})

install(DIRECTORY ${CIRCLE_SCHEMA_PYTHON_DIR}
        FILE_PERMISSIONS OWNER_WRITE OWNER_READ
                          GROUP_READ
                          WORLD_READ
        DESTINATION bin)
