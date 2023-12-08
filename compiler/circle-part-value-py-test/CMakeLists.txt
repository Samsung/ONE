if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

unset(RECIPE_LIST)
unset(PARTITION_LIST)
unset(OUTPUT_COUNT_LIST)
unset(TEST_DEPS)

macro(add RECIPE_NAME PARTITION_NAME OUTPUT_COUNT)
  list(APPEND RECIPE_LIST ${RECIPE_NAME})
  list(APPEND PARTITION_LIST ${PARTITION_NAME})
  list(APPEND OUTPUT_COUNT_LIST ${OUTPUT_COUNT})
endmacro(add)

# Read "test.lst"
include("test.lst")

list(LENGTH RECIPE_LIST RECIPE_LENGTH)
math(EXPR RECIPE_LENGTH_M1 "${RECIPE_LENGTH} - 1")

foreach(IDX RANGE ${RECIPE_LENGTH_M1})
  list(GET RECIPE_LIST ${IDX} RECIPE_NAME)
  list(GET PARTITION_LIST ${IDX} PARTITION_NAME)
  list(GET OUTPUT_COUNT_LIST ${IDX} OUTPUT_COUNT)

  # NOTE about the name:
  # Use '.recipe' name for source tflite and circle files
  # Use '.part' name for actual test folder and test files

  # Output to a folder
  set(PARTITIONER_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${PARTITION_NAME}")

  add_custom_command(OUTPUT ${PARTITIONER_OUTPUT_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PARTITIONER_OUTPUT_PATH}"
    COMMENT "Make directory ${PARTITIONER_OUTPUT_PATH}"
  )

  # Copy tflite
  set(TFLITE_SRC_PATH "${ARTIFACTS_BIN_PATH}/${RECIPE_NAME}.tflite")
  set(TFLITE_DST_PATH "${PARTITIONER_OUTPUT_PATH}/${PARTITION_NAME}.tflite")

  add_custom_command(OUTPUT ${TFLITE_DST_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy "${TFLITE_SRC_PATH}" "${TFLITE_DST_PATH}"
    DEPENDS ${TFLITE_SRC_PATH}
    COMMENT "Copy ${RECIPE_NAME}.tflite"
  )
  list(APPEND TEST_DEPS ${TFLITE_DST_PATH})

  # Copy circle
  set(CIRCLE_SRC_PATH "${ARTIFACTS_BIN_PATH}/${RECIPE_NAME}.circle")
  set(CIRCLE_DST_PATH "${PARTITIONER_OUTPUT_PATH}/${PARTITION_NAME}.circle")

  add_custom_command(OUTPUT ${CIRCLE_DST_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy "${CIRCLE_SRC_PATH}" "${CIRCLE_DST_PATH}"
    DEPENDS ${CIRCLE_SRC_PATH}
    COMMENT "Copy ${RECIPE_NAME}.circle"
  )
  list(APPEND TEST_DEPS ${CIRCLE_DST_PATH})

  # Copy .part
  set(PART_FILE "${PARTITION_NAME}.part")
  set(PART_SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/parts/${PART_FILE}")
  set(PART_DST_PATH "${PARTITIONER_OUTPUT_PATH}/${PART_FILE}")

  add_custom_command(OUTPUT ${PART_DST_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy "${PART_SRC_PATH}" "${PART_DST_PATH}"
    DEPENDS ${PART_SRC_PATH}
    COMMENT "Copy ${PART_FILE}"
  )
  list(APPEND TEST_DEPS ${PART_DST_PATH})

  # Partition connection file to generate
  set(PARTITIONER_CONN_JSON "${PARTITIONER_OUTPUT_PATH}/${PARTITION_NAME}.conn.json")

  # Run partitioner
  add_custom_command(OUTPUT ${PARTITIONER_CONN_JSON}
    COMMAND circle-partitioner "--part_file" "${PART_FILE}" "--input_file"
            "${PARTITION_NAME}.circle" "--work_path" "${PARTITIONER_OUTPUT_PATH}"
    DEPENDS circle-partitioner ${PART_DST_PATH} ${CIRCLE_DST_PATH}
    COMMENT "Parition ${RECIPE_NAME}.circle with ${PART_FILE}"
  )
  list(APPEND TEST_DEPS ${PARTITIONER_CONN_JSON})

  # Write .excnt file; expected count of output models
  set(COUNT_FILE "${PARTITION_NAME}.excnt")
  set(COUNT_FILE_PATH "${PARTITIONER_OUTPUT_PATH}/${COUNT_FILE}")
  add_custom_command(OUTPUT ${COUNT_FILE_PATH}
    COMMAND echo ${OUTPUT_COUNT} > ${COUNT_FILE_PATH}
    DEPENDS ${PART_SRC_PATH} ${PARTITIONER_OUTPUT_PATH}
    COMMENT "Write ${COUNT_FILE} with ${OUTPUT_COUNT}"
  )
  list(APPEND TEST_DEPS ${COUNT_FILE_PATH})
endforeach(IDX)

add_custom_target(circle_part_value_py_test_prepare ALL DEPENDS ${TEST_DEPS})
add_dependencies(circle_part_value_py_test_prepare common_artifacts_deps)

set(VIRTUALENV "${NNCC_OVERLAY_DIR}/venv_2_12_1")
set(TEST_LIST_FILE "test.lst")

add_test(NAME circle_part_value_py_test
  COMMAND ${VIRTUALENV}/bin/python -m pytest -sv test_circle_part_value.py
        --test_list ${TEST_LIST_FILE}
        --bin_dir ${CMAKE_CURRENT_BINARY_DIR}
        --circle_part_driver $<TARGET_FILE:circle_part_driver>
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
