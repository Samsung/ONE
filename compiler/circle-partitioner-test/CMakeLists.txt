# NOTE Test below are for circle-partitioner is partitioning itself.
#      Once this test passes, add partition to 'circle-part-value-test' for
#      full test.
if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

unset(RECIPE_LIST)
unset(PART_LIST)
unset(TEST_DEPS)

macro(add RECIPE_NAME PART_NAME)
  list(APPEND RECIPE_LIST ${RECIPE_NAME})
  list(APPEND PART_LIST ${PART_NAME})
endmacro(add)

# Read "test.lst"
include("test.lst")

list(LENGTH RECIPE_LIST RECIPE_LENGTH)
math(EXPR RECIPE_LENGTH_M1 "${RECIPE_LENGTH} - 1")

foreach(IDX RANGE ${RECIPE_LENGTH_M1})
  list(GET RECIPE_LIST ${IDX} RECIPE_NAME)
  list(GET PART_LIST ${IDX} PART_NAME)

  set(PART_OUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${PART_NAME}")

  add_custom_command(OUTPUT ${PART_OUT_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PART_OUT_PATH}"
    COMMENT "Make directory ${PART_OUT_PATH}"
  )

  set(CIRCLE_SRC_PATH "${ARTIFACTS_BIN_PATH}/${RECIPE_NAME}.circle")
  set(CIRCLE_DST_PATH "${PART_OUT_PATH}/${PART_NAME}.circle")

  # Copy circle
  add_custom_command(OUTPUT ${CIRCLE_DST_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy "${CIRCLE_SRC_PATH}" "${CIRCLE_DST_PATH}"
    DEPENDS ${CIRCLE_SRC_PATH}
    COMMENT "Copy ${RECIPE_NAME}.circle"
  )

  set(PART_FILE "${PART_NAME}.part")
  set(PART_SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/parts/${PART_FILE}")
  set(PART_DST_PATH "${PART_OUT_PATH}/${PART_FILE}")

  # Copy .part
  add_custom_command(OUTPUT ${PART_DST_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy "${PART_SRC_PATH}" "${PART_DST_PATH}"
    DEPENDS ${PART_SRC_PATH}
    COMMENT "Copy ${PART_FILE}"
  )

  # Run partitioner
  set(PART_CONN_JSON "${PART_OUT_PATH}/${PART_NAME}.conn.json")
  add_custom_command(OUTPUT ${PART_CONN_JSON}
    COMMAND circle_partitioner "${PART_FILE}" "${PART_NAME}.circle" "${PART_OUT_PATH}"
    DEPENDS circle_partitioner ${CIRCLE_DST_PATH} ${PART_DST_PATH}
    COMMENT "Parition ${RECIPE_NAME}.circle with ${PART_FILE}"
  )
  # NOTE this is checked in build time and not added with 'add_test' command
  # to reduce scripts to run testing. actual testing is done in 'circle-part-evel'

  list(APPEND TEST_DEPS ${CIRCLE_DST_PATH} ${PART_DST_PATH} ${PART_CONN_JSON})
endforeach(IDX)

add_custom_target(circle_partitioner_test ALL DEPENDS ${TEST_DEPS})
add_dependencies(circle_partitioner_test common_artifacts_deps)
