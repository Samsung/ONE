if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_include(TargetRequire)

unset(REQUIRED_TARGETS)
list(APPEND REQUIRED_TARGETS circle-inspect)
list(APPEND REQUIRED_TARGETS circle-verify)
list(APPEND REQUIRED_TARGETS circle_resizer)
list(APPEND REQUIRED_TARGETS dredd_rule_lib)
TargetRequire_Return(${REQUIRED_TARGETS})

unset(TEST_DEPS)
unset(TEST_NAMES)

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

set(oneValueArgs NEW_INPUTS_SIZES)
set(multiValueArgs "")

macro(Add RECIPE)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(NEW_INPUTS_SIZES "")
  if(ARG_NEW_INPUTS_SIZES)
    set(NEW_INPUTS_SIZES "--input_shapes" "${ARG_NEW_INPUTS_SIZES}")
  endif()

  set(CIRCLE_PATH "${ARTIFACTS_BIN_PATH}/${RECIPE}.circle")
  set(RESIZED_CIRCLE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${RECIPE}.resized.circle")

  # Generate resized .circle
  add_custom_command(OUTPUT ${RESIZED_CIRCLE_PATH}
    COMMAND $<TARGET_FILE:circle_resizer> --input_path ${CIRCLE_PATH} --output_path ${RESIZED_CIRCLE_PATH} ${NEW_INPUTS_SIZES}
    DEPENDS
      circle_resizer
      ${CIRCLE_PATH}
    COMMENT "Generate ${RECIPE}.resized.circle"
  )

  list(APPEND TEST_DEPS ${RESIZED_CIRCLE_PATH})
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
  COMMAND ${CMAKE_COMMAND} -E echo 'CIRCLE_RESIZER_PATH=\"$<TARGET_FILE:circle_resizer>\"' >> ${TEST_CONFIG}
  DEPENDS
    circle-inspect
    circle-verify
    circle_resizer
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
add_custom_target(circle-resizer_dredd_recipe_test ALL DEPENDS ${TEST_DEPS})
add_dependencies(circle-resizer_dredd_recipe_test common_artifacts_deps)

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

# Run tests
add_test(
  NAME resizer_dredd_recipe_test
  COMMAND "${TEST_RUNNER}"
          "${TEST_CONFIG}"
          "${ARTIFACTS_BIN_PATH}"
          ${TEST_NAMES}
)
