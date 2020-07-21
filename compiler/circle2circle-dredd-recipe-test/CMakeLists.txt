nnas_include(TargetRequire)

unset(REQUIRED_TARGETS)
list(APPEND REQUIRED_TARGETS circle-inspect)
list(APPEND REQUIRED_TARGETS circle-verify)
list(APPEND REQUIRED_TARGETS circle2circle)
list(APPEND REQUIRED_TARGETS dredd_rule_lib)
list(APPEND REQUIRED_TARGETS testDataGenerator)
TargetRequire_Return(${REQUIRED_TARGETS})

unset(TEST_DEPS)
unset(TEST_NAMES)

set(options "")
set(oneValueArgs "")
set(multiValueArgs PASS)

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

macro(Add RECIPE)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  unset(OPT_OPTIONS)
  foreach(src ${ARG_PASS})
    # option = "--${src}"
    list(APPEND OPT_OPTIONS "--${src}")
  endforeach(src ${ARG_PASS})

  set(CIRCLE_FILE "${RECIPE}.circle")
  set(CIRCLE_PATH "${ARTIFACTS_BIN_PATH}/${CIRCLE_FILE}")

  set(OPT_CIRCLE_FILE "${RECIPE}.opt.circle")
  set(OPT_CIRCLE_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${OPT_CIRCLE_FILE}")

  # Generate optimized .circle
  add_custom_command(OUTPUT ${OPT_CIRCLE_OUTPUT_PATH}
    COMMAND $<TARGET_FILE:circle2circle> ${OPT_OPTIONS} ${CIRCLE_PATH} ${OPT_CIRCLE_OUTPUT_PATH}
    DEPENDS $<TARGET_FILE:circle2circle> ${CIRCLE_PATH}
    COMMENT "Generate ${OPT_CIRCLE_FILE}"
  )

  list(APPEND TEST_DEPS ${OPT_CIRCLE_OUTPUT_PATH})
  list(APPEND TEST_NAMES ${RECIPE})
endmacro(Add)

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
  DEPENDS 
    circle-inspect
    circle-verify
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
add_custom_target(circle2circle_dredd_recipe_test ALL DEPENDS ${TEST_DEPS})

# Run tests
add_test(
  NAME circle2circle_dredd_recipe_test
  COMMAND "${TEST_RUNNER}"
          "${TEST_CONFIG}"
          "${CMAKE_CURRENT_BINARY_DIR}"
          ${TEST_NAMES}
)
