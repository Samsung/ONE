if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

unset(TEST_DEPS)
unset(CIRCLE_INTERPRETER_TESTS)

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

macro(addeval NAME)
  list(APPEND CIRCLE_INTERPRETER_TESTS ${NAME})
endmacro(addeval)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

add_custom_target(circle_interpreter_test_files ALL DEPENDS ${TEST_DEPS})
add_dependencies(circle_interpreter_test_files common_artifacts_deps)

add_test(NAME circle_interpreter_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/eval_driver.sh"
          "${CMAKE_CURRENT_BINARY_DIR}"
          "${ARTIFACTS_BIN_PATH}"
          "${NNCC_OVERLAY_DIR}/venv_2_8_0"
          "$<TARGET_FILE:circle-interpreter>"
          ${CIRCLE_INTERPRETER_TESTS}
)
