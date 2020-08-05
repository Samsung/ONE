macro(addread NAME)
  list(APPEND DAILY_READ_TESTS ${NAME})
endmacro(addread)

macro(addwrite NAME)
  list(APPEND DAILY_WRITE_TESTS ${NAME})
endmacro(addwrite)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

add_test(NAME luci_unit_readtest
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/readverify.sh"
          "${ARTIFACTS_BIN_PATH}"
          "${CMAKE_CURRENT_BINARY_DIR}"
          "$<TARGET_FILE:luci_readtester>"
          ${DAILY_READ_TESTS}
)

add_test(NAME luci_unit_writetest
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/writeverify.sh"
          "${ARTIFACTS_BIN_PATH}"
          "${CMAKE_CURRENT_BINARY_DIR}"
          "$<TARGET_FILE:luci_writetester>"
          ${DAILY_WRITE_TESTS}
)
