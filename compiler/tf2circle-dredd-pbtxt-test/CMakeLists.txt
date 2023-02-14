nnas_include(TargetRequire)

unset(REQUIRED_TARGETS)
list(APPEND REQUIRED_TARGETS circle-inspect)
list(APPEND REQUIRED_TARGETS circle-verify)
list(APPEND REQUIRED_TARGETS tf2circle)
list(APPEND REQUIRED_TARGETS tfkit)
list(APPEND REQUIRED_TARGETS dredd_rule_lib)
TargetRequire_Return(${REQUIRED_TARGETS})

nncc_find_resource(TensorFlowTests)

set(MODEL_REPO "${TensorFlowTests_DIR}") # Where to find text models to test

unset(KEYS)
unset(DEPS)

#
# processing models in test.lst and test.local.lst
#
# Example)
#
#  Add(NET_0025 RULE test.rule)
#    -> Read test.pbtxt file under res/TensorFlowTests/NET_0025 and create "NET_0025.circle"
#       Then the circle is tested against rules in test.rule file.
#
macro(Add MODEL_DIR)

  set(ARG_OPTION)
  set(ARG_ONE_VALUE RULE) # rule file name
  set(ARG_MULTI_VALUE)
  cmake_parse_arguments(ARG "${ARG_OPTION}" "${ARG_ONE_VALUE}" "${ARG_MULTI_VALUE}" ${ARGN})

  if(NOT ARG_RULE)
    message(FATAL_ERROR "RULE is mandadatory arg")
  endif()

  set(RULE_FILENAME ${ARG_RULE})

  set(TARGET_TESTNAME "${MODEL_DIR}")

  set(MODEL_SOURCE_DIR "${MODEL_REPO}/${MODEL_DIR}")

  set(TXT_SOURCE_PBTXT_PATH "${MODEL_SOURCE_DIR}/test.pbtxt")
  set(TXT_SOURCE_INFO_PATH "${MODEL_SOURCE_DIR}/test.info")
  set(TXT_SOURCE_RULE_PATH "${MODEL_SOURCE_DIR}/${RULE_FILENAME}")

  set(TXT_TARGET_PB_PATH "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_TESTNAME}.pb")
  set(TXT_TARGET_PBTXT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_TESTNAME}.pbtxt")
  set(TXT_TARGET_INFO_PATH "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_TESTNAME}.info")
  set(TXT_TARGET_RULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_TESTNAME}.rule")

  if(NOT EXISTS "${TXT_SOURCE_PBTXT_PATH}")
    message(FATAL_ERROR "${TXT_SOURCE_PBTXT_PATH} - pbtxt file does not exist")
  endif(NOT EXISTS "${TXT_SOURCE_PBTXT_PATH}")

  if(NOT EXISTS "${TXT_SOURCE_INFO_PATH}")
    message(FATAL_ERROR "${TXT_SOURCE_INFO_PATH} - info file does not exist")
  endif(NOT EXISTS "${TXT_SOURCE_INFO_PATH}")

  if(NOT EXISTS "${TXT_SOURCE_RULE_PATH}")
    message(FATAL_ERROR "${TXT_SOURCE_RULE_PATH} - rule file does not exist")
  endif(NOT EXISTS "${TXT_SOURCE_RULE_PATH}")

  # Copy .pbtxt
  add_custom_command(
    OUTPUT ${TXT_TARGET_PBTXT_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy "${TXT_SOURCE_PBTXT_PATH}" "${TXT_TARGET_PBTXT_PATH}"
    DEPENDS ${TXT_SOURCE_PBTXT_PATH}
    COMMENT "Generate ${TXT_TARGET_PBTXT_PATH}")

  # Copy .info
  add_custom_command(
    OUTPUT ${TXT_TARGET_INFO_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy "${TXT_SOURCE_INFO_PATH}" "${TXT_TARGET_INFO_PATH}"
    DEPENDS ${TXT_SOURCE_INFO_PATH}
    COMMENT "Generate ${TXT_TARGET_INFO_PATH}")

  # Copy .rule
  add_custom_command(
    OUTPUT ${TXT_TARGET_RULE_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy "${TXT_SOURCE_RULE_PATH}" "${TXT_TARGET_RULE_PATH}"
    DEPENDS ${TXT_SOURCE_RULE_PATH}
    COMMENT "Generate ${TXT_TARGET_RULE_PATH}")

  # Generate .pb from .pbtxt
  add_custom_command(
    OUTPUT ${TXT_TARGET_PB_PATH}
    COMMAND $<TARGET_FILE:tfkit> encode ${TXT_TARGET_PBTXT_PATH} ${TXT_TARGET_PB_PATH}
    DEPENDS ${TXT_TARGET_PBTXT_PATH}
    COMMENT "Generate ${TXT_TARGET_PB_PATH}")

  # Generate .test file which declares path of target pb, info, rule files
  # this file is used inside runner.sh
  set(TEST_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_TESTNAME}.test")

  add_custom_command(
    OUTPUT ${TEST_CONFIG_FILE}
    COMMAND ${CMAKE_COMMAND} -E remove -f ${TEST_CONFIG_FILE}
    COMMAND ${CMAKE_COMMAND} -E echo 'MODEL_PB_PATH="${TXT_TARGET_PB_PATH}"' >> ${TEST_CONFIG_FILE}
    COMMAND ${CMAKE_COMMAND} -E echo 'MODEL_INFO_PATH="${TXT_TARGET_INFO_PATH}"' >> ${TEST_CONFIG_FILE}
    COMMAND ${CMAKE_COMMAND} -E echo 'MODEL_RULE_PATH="${TXT_TARGET_RULE_PATH}"' >> ${TEST_CONFIG_FILE}
    DEPENDS ${TXT_TARGET_PB_PATH} ${TXT_TARGET_INFO_PATH} ${TXT_TARGET_RULE_PATH}
    COMMENT "Generate ${TARGET_TESTNAME} configuration for TXT")

  list(APPEND DEPS "${TEST_CONFIG_FILE}")
  list(APPEND KEYS "${TARGET_TESTNAME}")

endmacro(Add)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

#
# Generate toolchain.config
#
set(TOOLCHAIN_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/toolchain.config")

add_custom_command(
  OUTPUT ${TOOLCHAIN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E remove -f ${TOOLCHAIN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'CIRCLE_INSPECT_PATH=\"$<TARGET_FILE:circle-inspect>\"' >> ${TOOLCHAIN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'CIRCLE_VERIFY_PATH=\"$<TARGET_FILE:circle-verify>\"' >> ${TOOLCHAIN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'TF2CIRCLE_PATH=\"$<TARGET_FILE:tf2circle>\"' >> ${TOOLCHAIN_CONFIG}
  # add more if new excutable file is needed in runner.sh and rule-lib.sh
  DEPENDS circle-inspect circle-verify tf2circle
  COMMENT "Generate toolchin configuration")

list(APPEND DEPS "${TOOLCHAIN_CONFIG}")

#
# copy runner.sh
#
set(SOURCE_RUNNER "${CMAKE_CURRENT_SOURCE_DIR}/runner.sh")
set(TARGET_RUNNER "${CMAKE_CURRENT_BINARY_DIR}/runner.sh")

add_custom_command(
  OUTPUT ${TARGET_RUNNER}
  COMMAND ${CMAKE_COMMAND} -E copy "${SOURCE_RUNNER}" "${TARGET_RUNNER}"
  DEPENDS ${SOURCE_RUNNER}
  COMMENT "Generate test runner")

list(APPEND DEPS "${TARGET_RUNNER}")

#
# copy rule-lib.sh (a library of shell script functions)
#

# getting path for rule-lib.sh in dredd-rule-lib
get_target_property(DREDD_RULE_LIB_DIR dredd_rule_lib BINARY_DIR)

set(SOURCE_RULE_LIB "${DREDD_RULE_LIB_DIR}/rule-lib.sh")
set(TARGET_RULE_LIB "${CMAKE_CURRENT_BINARY_DIR}/rule-lib.sh")

add_custom_command(
  OUTPUT ${TARGET_RULE_LIB}
  COMMAND ${CMAKE_COMMAND} -E copy "${SOURCE_RULE_LIB}" "${TARGET_RULE_LIB}"
  DEPENDS ${SOURCE_RULE_LIB}
  COMMENT "Generate rule lib")

list(APPEND DEPS "${TARGET_RULE_LIB}")

# Generate dependencies
add_custom_target(tf2circle_dredd_pbtxt_deps ALL DEPENDS ${DEPS})

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

add_test(NAME tf2circle_dredd_pbtxt_test COMMAND "${TARGET_RUNNER}" "${TOOLCHAIN_CONFIG}" ${KEYS})
