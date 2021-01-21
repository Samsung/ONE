unset(TEST_DEPS)
unset(LUCI_PASS_VALUE_TESTS)

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

macro(addeval RECIPE PASS_OPTION)
  list(APPEND LUCI_PASS_VALUE_TESTS ${RECIPE})

  set(CIRCLE_FILE "${RECIPE}.circle")
  set(CIRCLE_PATH "${ARTIFACTS_BIN_PATH}/${CIRCLE_FILE}")

  set(PASS_CIRCLE_FILE "${RECIPE}.pass.circle")
  set(PASS_CIRCLE_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${PASS_CIRCLE_FILE}")

  set(DASH_PASS_OPTION "--${PASS_OPTION}")

  # Generate optimized .circle
  add_custom_command(OUTPUT ${PASS_CIRCLE_OUTPUT_PATH}
    COMMAND $<TARGET_FILE:circle2circle> ${DASH_PASS_OPTION} ${CIRCLE_PATH} ${PASS_CIRCLE_OUTPUT_PATH}
    DEPENDS $<TARGET_FILE:circle2circle> ${CIRCLE_PATH}
    COMMENT "Generate ${PASS_CIRCLE_FILE} with ${DASH_PASS_OPTION}"
  )

  # depends
  list(APPEND TEST_DEPS ${PASS_CIRCLE_OUTPUT_PATH})

endmacro(addeval)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

add_custom_target(luci_pass_value_test_files ALL DEPENDS ${TEST_DEPS})
add_dependencies(luci_pass_value_test_files common_artifacts_deps)

add_test(NAME luci_pass_value_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/eval_driver.sh"
          "${CMAKE_CURRENT_BINARY_DIR}"
          "${ARTIFACTS_BIN_PATH}"
          "${NNCC_OVERLAY_DIR}/venv_2_3_0"
          "$<TARGET_FILE:luci_eval_tester>"
          ${LUCI_PASS_VALUE_TESTS}
)
