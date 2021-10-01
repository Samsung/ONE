unset(CIRCLE_ORIGIN)
unset(CIRCLE_SELECTED)
unset(NODES_LIST)

macro(addeval NAME)
  list(APPEND CIRCLE_OPSELECTOR_TEST ${NAME})
endmacro(addeval)

# Read "test.lst"
include("test.lst")
include("test.local.lst" OPTIONAL)

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

add_test(NAME circle_opselector_arg_validity_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/arg_validity_check.sh"
)

add_test(NAME circle_opselector_value_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/compare_circles.sh"
          "${CMAKE_CURRENT_BINARY_DIR}"
          "${ARTIFACTS_BIN_PATH}"
          "${NNCC_OVERLAY_DIR}/venv_2_3_0"
          "$<TARGET_FILE:luci_eval_driver>"
          "$<TARGET_FILE:circle-opselector>"
          "${CIRCLE_OPSELECTOR_TEST}"
)
