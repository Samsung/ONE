unset(LUCI_VALUE_TESTS)

macro(addeval NAME)
  list(APPEND LUCI_VALUE_TESTS ${NAME})
endmacro(addeval)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

# Generate dependencies
add_custom_target(luci_eval_testfiles ALL DEPENDS ${TESTFILES})

add_subdirectory(tester)

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

add_test(NAME luci_value_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/evalverify.sh"
          "${CMAKE_CURRENT_BINARY_DIR}"
          "${ARTIFACTS_BIN_PATH}"
          ${LUCI_VALUE_TESTS}
)
