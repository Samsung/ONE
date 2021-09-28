unset(CIRCLE_ORIGIN)
unset(CIRCLE_SELECTED)
unset(NODES_LIST)

macro(addeval CIRCLE1 CIRCLE2 NODES)
  list(APPEND CIRCLE_ORIGIN ${CIRCLE1})
  list(APPEND CIRCLE_SELECTED ${CIRCLE2})
  list(APPEND NODES_LIST ${NODES})
endmacro(addeval)

# Read "test.lst"
include("test.lst")
include("test.local.lst" OPTIONAL)

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

add_test(NAME arg_validity_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/arg_validity_check.sh"
)

add_test(NAME circle_opselector_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/compare_circles.sh"
          "${CIRCLE_ORIGIN}"
          "${CIRCLE_SELECTED}"
          "${NODES_LIST}"
)
