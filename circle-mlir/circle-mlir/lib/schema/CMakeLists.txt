include(UseFlatbuffers)

set(SCHEMA_FILE "${RES_CIRCLE_SCHEMA}/0.9/circle_schema.fbs")
set(SCHEMA_TARGET "${CMAKE_CURRENT_BINARY_DIR}/schema.fbs")

add_custom_command(OUTPUT "${SCHEMA_TARGET}"
  COMMAND ${CMAKE_COMMAND} -E copy "${SCHEMA_FILE}" "${SCHEMA_TARGET}"
  DEPENDS "${SCHEMA_FILE}"
  COMMENT "lib/schema: copy schema.fbs"
)
list(APPEND DEPS "${CMAKE_CURRENT_BINARY_DIR}/schema.fbs")
add_custom_target(circle_schema_deps ALL DEPENDS ${DEPS})

FlatBuffers_Target(circle_schema
  OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/circle_schema/"
  INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  SCHEMA_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  SCHEMA_FILES "schema.fbs"
)
