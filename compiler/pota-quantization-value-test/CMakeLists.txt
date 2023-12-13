if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

unset(QUANTIZATION_VALUE_TEST)
unset(QUANTIZATION_VALUE_TEST_WITH_PARAM)
unset(QUANTIZATION_CONFIG_VALUE_TEST)
unset(QUANTIZATION_CONFIG_VALUE_TEST_WITH_PARAM)
unset(QUANTIZATION_WO_VALUE_TEST_WITH_PARAM)

macro(addTest NAME GRANULARITY DTYPE)
  list(APPEND QUANTIZATION_VALUE_TEST ${NAME})
  list(APPEND QUANTIZATION_VALUE_TEST_WITH_PARAM ${NAME} ${GRANULARITY} ${DTYPE})
endmacro(addTest)

macro(addQConfTest NAME GRANULARITY DTYPE)
  list(APPEND QUANTIZATION_CONFIG_VALUE_TEST ${NAME})
  list(APPEND QUANTIZATION_CONFIG_VALUE_TEST_WITH_PARAM ${NAME} ${GRANULARITY} ${DTYPE})
endmacro(addQConfTest)

macro(addWeightsOnlyTest NAME GRANULARITY DTYPE)
    list(APPEND QUANTIZATION_WO_VALUE_TEST_WITH_PARAM ${NAME} ${GRANULARITY} ${DTYPE})
endmacro(addWeightsOnlyTest)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

unset(TEST_DEPS)

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/gen_h5_explicit_inputs_all.py"
               "${CMAKE_CURRENT_BINARY_DIR}/gen_h5_explicit_inputs_all.py" COPYONLY)

set(VIRTUALENV "${NNCC_OVERLAY_DIR}/venv_2_12_1")

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
  COMMAND ${CMAKE_COMMAND} -E echo 'VIRTUALENV=\"${VIRTUALENV}\"' >> ${TEST_CONFIG}
  DEPENDS record-minmax
  DEPENDS circle-quantizer
  DEPENDS circle-tensordump
  COMMENT "Generate test configuration"
)

# Import pics module
get_target_property(PICS_BIN_PATH pics BINARY_DIR)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/circle
                   COMMAND ${CMAKE_COMMAND} -E create_symlink
                   ${PICS_BIN_PATH}/circle ${CMAKE_CURRENT_BINARY_DIR}/circle)

list(APPEND TEST_DEPS "${TEST_CONFIG}" "${CMAKE_CURRENT_BINARY_DIR}/circle")

# This enforces CMake to generate all the dependencies during "build" phase
add_custom_target(pota_quantization_value_test_deps ALL DEPENDS ${TEST_DEPS})

# Run tests
# NOTE. The order of the test matters as some tests depend on previous tests.
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

add_test(
  NAME pota_fake_wquant_test_with_config
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/test_fake_wquant_with_config.sh"
          "${TEST_CONFIG}"
          "${ARTIFACTS_BIN_PATH}"
          ${QUANTIZATION_CONFIG_VALUE_TEST_WITH_PARAM}
)

add_test(
  NAME pota_parallel_record_minmax_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/test_parallel_record_minmax.sh"
          "${TEST_CONFIG}"
          "${ARTIFACTS_BIN_PATH}"
          ${QUANTIZATION_VALUE_TEST_WITH_PARAM}
)

add_test(
  NAME pota_quantization_test_with_config
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/test_quantization_with_config.sh"
          "${TEST_CONFIG}"
          "${ARTIFACTS_BIN_PATH}"
          ${QUANTIZATION_CONFIG_VALUE_TEST_WITH_PARAM}
)

add_test(
  NAME pota_wo_quantization_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/test_wo_quantization.sh"
          "${TEST_CONFIG}"
          "${ARTIFACTS_BIN_PATH}"
          ${QUANTIZATION_WO_VALUE_TEST_WITH_PARAM}
)

set_tests_properties(pota_record_minmax_test PROPERTIES DEPENDS pota_fake_wquant_test)
set_tests_properties(pota_quantization_test PROPERTIES DEPENDS pota_record_minmax_test)
set_tests_properties(pota_parallel_record_minmax_test PROPERTIES DEPENDS pota_record_minmax_test)
set_tests_properties(pota_quantization_test_with_config PROPERTIES DEPENDS pota_fake_wquant_test_with_config)
