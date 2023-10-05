if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

unset(Q_IMPLANT_TESTS)

macro(addeval NAME)
  list(APPEND Q_IMPLANT_TESTS ${NAME})
endmacro(addeval)

include("test.lst")

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

add_test(NAME q_implant_op_level_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/q-implant-op-level-test.sh"
          "${CMAKE_CURRENT_BINARY_DIR}"
          "${ARTIFACTS_BIN_PATH}"
          "${NNCC_OVERLAY_DIR}/venv_2_12_1"
          "$<TARGET_FILE:q-implant>"
          "$<TARGET_FILE:circle-tensordump>"
          ${Q_IMPLANT_TESTS}
)
