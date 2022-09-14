if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

unset(CIRCLE_INTERPRETER_TESTS)

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)


unset(TEST_ARGS_LIST)
# Macro for command line circle-interpreter validation test
# Example of usage
#   testcmd(POS_000 POS sample.circle sample.input sample.output)
#   testcmd(NEG_000 NEG not-existing.circle invalid-option)
macro(testcmd TEST_NAME TEST_TYPE)
  list(APPEND TEST_ARGS_LIST ${TEST_NAME})
  list(APPEND TEST_ARGS_LIST ${TEST_TYPE})

  set(EXTRA_ARGS ${ARGN})
  unset(CMD_STR)

  set(WORK_DIR ${ARTIFACTS_BIN_PATH})

  foreach(ARGUMENT ${EXTRA_ARGS})
    set(CMD_STR "${CMD_STR} ${ARGUMENT}")
  endforeach()
  
  list(APPEND TEST_ARGS_LIST ${CMD_STR})
endmacro(testcmd)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

add_test(NAME circle_interpreter_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/run_command.sh"
          "$<TARGET_FILE:circle-interpreter>"
          ${TEST_ARGS_LIST}
)
