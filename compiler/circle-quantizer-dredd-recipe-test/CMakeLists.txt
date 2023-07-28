if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_include(TargetRequire)

unset(REQUIRED_TARGETS)
list(APPEND REQUIRED_TARGETS circle-inspect)
list(APPEND REQUIRED_TARGETS circle-verify)
list(APPEND REQUIRED_TARGETS circle-quantizer)
list(APPEND REQUIRED_TARGETS record-minmax)
list(APPEND REQUIRED_TARGETS dredd_rule_lib)
TargetRequire_Return(${REQUIRED_TARGETS})

unset(TEST_DEPS)
unset(TEST_NAMES)

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

set(options USE_QCONFIG)
set(oneValueArgs DTYPE GRANULARITY INPUT_DTYPE OUTPUT_DTYPE)
set(multiValueArgs "")

macro(Add RECIPE)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(QCONFIG_OPT "")
  if(ARG_USE_QCONFIG)
    set(QCONFIG_OPT "--config" "${ARTIFACTS_BIN_PATH}/${RECIPE}.qconf.json")
  endif()

  set(INPUT_DTYPE_OPT "")
  if(ARG_INPUT_DTYPE)
    set(INPUT_DTYPE_OPT "--input_type" "${ARG_INPUT_DTYPE}")
  endif()

  set(OUTPUT_DTYPE_OPT "")
  if(ARG_OUTPUT_DTYPE)
    set(OUTPUT_DTYPE_OPT "--output_type" "${ARG_OUTPUT_DTYPE}")
  endif()

  set(CIRCLE_PATH "${ARTIFACTS_BIN_PATH}/${RECIPE}.circle")
  set(FAKE_QUANT_CIRCLE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${RECIPE}.fq.circle")
  set(RECORDED_CIRCLE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${RECIPE}.recorded.circle")
  set(QUANT_CIRCLE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${RECIPE}.q.circle")

  # Generate quantized .circle
  add_custom_command(OUTPUT ${QUANT_CIRCLE_PATH}
    COMMAND $<TARGET_FILE:circle-quantizer> --quantize_dequantize_weights float32 ${ARG_DTYPE} ${ARG_GRANULARITY} ${QCONFIG_OPT} ${CIRCLE_PATH} ${FAKE_QUANT_CIRCLE_PATH}
    COMMAND $<TARGET_FILE:record-minmax> --input_model ${FAKE_QUANT_CIRCLE_PATH} --output_model ${RECORDED_CIRCLE_PATH}
    COMMAND $<TARGET_FILE:circle-quantizer>
      --quantize_with_minmax float32 ${ARG_DTYPE} ${ARG_GRANULARITY}
      ${QCONFIG_OPT} ${RECORDED_CIRCLE_PATH} ${QUANT_CIRCLE_PATH}
      ${INPUT_DTYPE_OPT} ${OUTPUT_DTYPE_OPT}
    DEPENDS 
      circle-quantizer
      record-minmax
      ${CIRCLE_PATH}
    COMMENT "Generate ${RECIPE}.q.circle"
  )

  list(APPEND TEST_DEPS ${QUANT_CIRCLE_PATH})
  list(APPEND TEST_NAMES ${RECIPE})
endmacro(Add)

# Macro to generate fully fake-quantized models
macro(AddFakeQuant RECIPE)
  set(CIRCLE_PATH "${ARTIFACTS_BIN_PATH}/${RECIPE}.circle")
  # NOTE We use .q.circle because it is convention for output file (see testall.sh for more details)
  set(FULL_FAKE_QUANT_CIRCLE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${RECIPE}.q.circle")

  # Generate fully fake-quantized .circle
  add_custom_command(OUTPUT ${FULL_FAKE_QUANT_CIRCLE_PATH}
    COMMAND $<TARGET_FILE:circle-quantizer> --fake_quantize ${CIRCLE_PATH} ${FULL_FAKE_QUANT_CIRCLE_PATH}
    DEPENDS
      circle-quantizer
      ${CIRCLE_PATH}
    COMMENT "Generate ${RECIPE}.q.circle"
  )

  list(APPEND TEST_DEPS ${FULL_FAKE_QUANT_CIRCLE_PATH})
  list(APPEND TEST_NAMES ${RECIPE})
endmacro(AddFakeQuant)

# Macro to generate re-quantized models
macro(AddReQuant RECIPE)
  set(CIRCLE_PATH "${ARTIFACTS_BIN_PATH}/${RECIPE}.circle")
  # NOTE We use .q.circle because it is convention for output file (see testall.sh for more details)
  set(REQUANT_CIRCLE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${RECIPE}.q.circle")

  # Generate re-quantized .circle
  add_custom_command(OUTPUT ${REQUANT_CIRCLE_PATH}
    COMMAND $<TARGET_FILE:circle-quantizer> --requantize int8 uint8 ${CIRCLE_PATH} ${REQUANT_CIRCLE_PATH}
    DEPENDS
      circle-quantizer
      ${CIRCLE_PATH}
    COMMENT "Generate ${RECIPE}.q.circle"
  )

  list(APPEND TEST_DEPS ${REQUANT_CIRCLE_PATH})
  list(APPEND TEST_NAMES ${RECIPE})
endmacro(AddReQuant)

# Macro to quantize without quantize_dequantize_weights
macro(AddSkipQDQW RECIPE)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(QCONFIG_OPT "")
  if(ARG_USE_QCONFIG)
    set(QCONFIG_OPT "--config" "${ARTIFACTS_BIN_PATH}/${RECIPE}.qconf.json")
  endif()

  set(INPUT_DTYPE_OPT "")
  if(ARG_INPUT_DTYPE)
    set(INPUT_DTYPE_OPT "--input_type" "${ARG_INPUT_DTYPE}")
  endif()

  set(OUTPUT_DTYPE_OPT "")
  if(ARG_OUTPUT_DTYPE)
    set(OUTPUT_DTYPE_OPT "--output_type" "${ARG_OUTPUT_DTYPE}")
  endif()

  set(CIRCLE_PATH "${ARTIFACTS_BIN_PATH}/${RECIPE}.circle")
  set(RECORDED_CIRCLE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${RECIPE}.recorded.circle")
  set(QUANT_CIRCLE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${RECIPE}.q.circle")

  # Generate quantized .circle
  add_custom_command(OUTPUT ${QUANT_CIRCLE_PATH}
    COMMAND $<TARGET_FILE:record-minmax> --input_model ${CIRCLE_PATH} --output_model ${RECORDED_CIRCLE_PATH}
    COMMAND $<TARGET_FILE:circle-quantizer>
      --quantize_with_minmax float32 ${ARG_DTYPE} ${ARG_GRANULARITY}
      ${QCONFIG_OPT} ${RECORDED_CIRCLE_PATH} ${QUANT_CIRCLE_PATH}
      ${INPUT_DTYPE_OPT} ${OUTPUT_DTYPE_OPT}
    DEPENDS
      circle-quantizer
      record-minmax
      ${CIRCLE_PATH}
    COMMENT "Generate ${RECIPE}.q.circle"
  )

  list(APPEND TEST_DEPS ${QUANT_CIRCLE_PATH})
  list(APPEND TEST_NAMES ${RECIPE})
endmacro(AddSkipQDQW)

# Read "test.lst"
include("test.lst")

##
## Copy testall
##
set(TEST_RUNNER "${CMAKE_CURRENT_BINARY_DIR}/testall.sh")
set(TEST_RUNNER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/testall.sh")

add_custom_command(
  OUTPUT ${TEST_RUNNER}
  COMMAND ${CMAKE_COMMAND} -E copy "${TEST_RUNNER_SOURCE}" "${TEST_RUNNER}"
  DEPENDS ${TEST_RUNNER_SOURCE}
  COMMENT "Generate test runner"
)

list(APPEND TEST_DEPS "${TEST_RUNNER}")

###
### Generate test.config
###
set(TEST_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/test.config")

add_custom_command(
  OUTPUT ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E remove -f ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'CIRCLE_INSPECT_PATH=\"$<TARGET_FILE:circle-inspect>\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'CIRCLE_VERIFY_PATH=\"$<TARGET_FILE:circle-verify>\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'RECORD_MINMAX_PATH=\"$<TARGET_FILE:record-minmax>\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'CIRCLE_QUANTIZER_PATH=\"$<TARGET_FILE:circle-quantizer>\"' >> ${TEST_CONFIG}
  DEPENDS
    circle-inspect
    circle-verify
    record-minmax
    circle-quantizer
  COMMENT "Generate test configuration"
)

list(APPEND TEST_DEPS "${TEST_CONFIG}")

#
# copy rule-lib.sh (a library of shell script functions)
#

# getting path for rule-lib.sh in dredd-rule-lib
get_target_property(DREDD_RULE_LIB_DIR dredd_rule_lib BINARY_DIR)

set(RULE_LIB_SOURCE_PATH "${DREDD_RULE_LIB_DIR}/rule-lib.sh")
set(RULE_LIB_BINARY_PATH "${CMAKE_CURRENT_BINARY_DIR}/rule-lib.sh")

add_custom_command(
  OUTPUT ${RULE_LIB_BINARY_PATH}
  COMMAND ${CMAKE_COMMAND} -E copy "${RULE_LIB_SOURCE_PATH}" "${RULE_LIB_BINARY_PATH}"
  DEPENDS ${RULE_LIB_SOURCE_PATH}
  COMMENT "Generate rule lib"
)

list(APPEND TEST_DEPS "${RULE_LIB_BINARY_PATH}")

# Generate dependencies
add_custom_target(circle_quantizer_dredd_recipe_test ALL DEPENDS ${TEST_DEPS})
add_dependencies(circle_quantizer_dredd_recipe_test common_artifacts_deps)

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

# Run tests
add_test(
  NAME circle_quantizer_dredd_recipe_test
  COMMAND ${TEST_RUNNER}
          ${TEST_CONFIG}
          ${ARTIFACTS_BIN_PATH}
          ${TEST_NAMES}
)
