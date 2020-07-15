unset(QUANTIZATION_VALUE_TEST)
unset(QUANTIZATION_VALUE_TEST_WITH_PARAM)

macro(addTest NAME GRANULARITY DTYPE)
  list(APPEND QUANTIZATION_VALUE_TEST ${NAME})
  list(APPEND QUANTIZATION_VALUE_TEST_WITH_PARAM ${NAME} ${GRANULARITY} ${DTYPE})
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
  COMMAND ${CMAKE_COMMAND} -E echo 'CIRCLE_QUANTIZER_PATH=\"$<TARGET_FILE:circle-quantizer>\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'CIRCLE_TENSORDUMP_PATH=\"$<TARGET_FILE:circle-tensordump>\"' >> ${TEST_CONFIG}
  DEPENDS record-minmax
  DEPENDS circle-quantizer
  DEPENDS circle-tensordump
  COMMENT "Generate test configuration"
)

list(APPEND TEST_DEPS "${TEST_CONFIG}")

# This enforces CMake to generate all the dependencies during "build" phase
add_custom_target(pota_quantization_value_test_deps ALL DEPENDS ${TEST_DEPS})

# Run tests
add_test(
  NAME pota_fake_wquant_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/test_fake_wquant.sh"
          "${TEST_CONFIG}"
          "${ARTIFACTS_BIN_PATH}"
          ${QUANTIZATION_VALUE_TEST_WITH_PARAM}
)

add_test(
  NAME pota_record_minmax_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/test_record_minmax.sh"
          "${TEST_CONFIG}"
          "${ARTIFACTS_BIN_PATH}"
          ${QUANTIZATION_VALUE_TEST_WITH_PARAM}
)

add_test(
  NAME pota_quantization_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/test_quantization.sh"
          "${TEST_CONFIG}"
          "${ARTIFACTS_BIN_PATH}"
          ${QUANTIZATION_VALUE_TEST_WITH_PARAM}
)

set_tests_properties(pota_record_minmax_test PROPERTIES DEPENDS pota_fake_wquant_test)
set_tests_properties(pota_quantization_test PROPERTIES DEPENDS pota_record_minmax_test)
