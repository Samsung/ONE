if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

unset(RECORD_MINMAX_CONVERSION_TEST)

macro(addTest NAME)
  list(APPEND RECORD_MINMAX_CONVERSION_TEST ${NAME})
endmacro(addTest)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

unset(TEST_DEPS)

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

###
### Generate test.config
###
set(TEST_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/test.config")

add_custom_command(
  OUTPUT ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E remove -f ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'RECORD_MINMAX_PATH=\"$<TARGET_FILE:record-minmax>\"' >> ${TEST_CONFIG}
  DEPENDS record-minmax
  COMMENT "Generate test configuration"
)

list(APPEND TEST_DEPS "${TEST_CONFIG}")

# This enforces CMake to generate all the dependencies during "build" phase
add_custom_target(record_minmax_conversion_test_deps ALL DEPENDS ${TEST_DEPS})

# Run tests
add_test(
  NAME record_minmax_conversion_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/testall.sh"
          "${TEST_CONFIG}"
          "${ARTIFACTS_BIN_PATH}"
          "${NNCC_OVERLAY_DIR}/venv_2_12_1"
          ${RECORD_MINMAX_CONVERSION_TEST}
)

if(ONE_UBUNTU_CODENAME_JAMMY)
  add_test(
    NAME record_minmax_conversion_210_test
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/testall.sh"
            "${TEST_CONFIG}"
            "${ARTIFACTS_BIN_PATH}"
            "${NNCC_OVERLAY_DIR}/venv_2_10_1"
            ${RECORD_MINMAX_CONVERSION_TEST}
  )
endif(ONE_UBUNTU_CODENAME_JAMMY)
