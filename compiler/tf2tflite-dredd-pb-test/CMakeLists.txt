if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_include(TargetRequire)

unset(REQUIRED_TARGETS)
list(APPEND REQUIRED_TARGETS tfl-inspect)
list(APPEND REQUIRED_TARGETS tfl-verify)
list(APPEND REQUIRED_TARGETS tf2tflite)
list(APPEND REQUIRED_TARGETS dredd_rule_lib)
TargetRequire_Return(${REQUIRED_TARGETS})

set(PB_MODEL_REPO "${CMAKE_CURRENT_SOURCE_DIR}/contrib") # Where to find models to test

unset(KEYS)
unset(DEPS)

function(check_file_exist)

  foreach(FILE_PATH IN LISTS ARGV)
    if(NOT EXISTS "${FILE_PATH}")
      message(FATAL_ERROR "${FILE_PATH} does not exist." )
    endif()
  endforeach()

endfunction()

#
# processing models in contrib.lst
#
# Example)
#
#  Add(Inception_v3 RULE tflite_1.0_rel_requirement.rule)
#    -> Read compiler/tf2tflite-dredd-pb-test/contrib/Inception_v3/model.pb and generate
#      "Inception_v3.tflite". Then rule file is tested for the generated tflite file.
#
macro(Add MODEL_DIR)

  set(ARG_OPTION)
  set(ARG_ONE_VALUE RULE) # rule file name
  set(ARG_MULTI_VALUE)
  cmake_parse_arguments(ARG "${ARG_OPTION}" "${ARG_ONE_VALUE}" "${ARG_MULTI_VALUE}" ${ARGN})

  if(NOT ARG_RULE )
    message( FATAL_ERROR "RULE is mandadatory arg" )
  endif()

  set(RULE_FILENAME ${ARG_RULE})

  set(MODEL_SOURCE_DIR "${PB_MODEL_REPO}/${MODEL_DIR}")

  set(PB_PATH "${MODEL_SOURCE_DIR}/model.pb")
  set(INFO_PATH "${MODEL_SOURCE_DIR}/model.info")
  set(RULE_PATH "${MODEL_SOURCE_DIR}/${RULE_FILENAME}")

  check_file_exist(${PB_PATH} ${INFO_PATH} ${RULE_PATH})

  # Generate .test file which declares path of target pb, info, rule files
  set(TARGET_TESTNAME "${MODEL_DIR}")
  set(TEST_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_TESTNAME}.test")

  add_custom_command(
    OUTPUT ${TEST_CONFIG_FILE}
    COMMAND ${CMAKE_COMMAND} -E remove -f ${TEST_CONFIG_FILE}
    COMMAND ${CMAKE_COMMAND} -E echo 'MODEL_PB_PATH="${PB_PATH}"' >> ${TEST_CONFIG_FILE}
    COMMAND ${CMAKE_COMMAND} -E echo 'MODEL_INFO_PATH="${INFO_PATH}"' >> ${TEST_CONFIG_FILE}
    COMMAND ${CMAKE_COMMAND} -E echo 'MODEL_RULE_PATH="${RULE_PATH}"' >> ${TEST_CONFIG_FILE}
    DEPENDS
      ${PB_PATH}
    COMMENT "Generate ${TARGET_TESTNAME} configuration for BIN"
  )

  list(APPEND KEYS "${TARGET_TESTNAME}")
  list(APPEND DEPS "${TEST_CONFIG_FILE}")

endmacro(Add)

include(contrib.lst OPTIONAL)

#
# Generate toolchain.config
#
set(TOOLCHAIN_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/toolchain.config")

add_custom_command(
  OUTPUT ${TOOLCHAIN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E remove -f ${TOOLCHAIN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'TFL_INSPECT_PATH=\"$<TARGET_FILE:tfl-inspect>\"' >> ${TOOLCHAIN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'TFL_VERIFY_PATH=\"$<TARGET_FILE:tfl-verify>\"' >> ${TOOLCHAIN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'TF2TFLITE_PATH=\"$<TARGET_FILE:tf2tflite>\"' >> ${TOOLCHAIN_CONFIG}
  # add more if new excutable file is needed in runner.sh and rule-lib.sh
  DEPENDS
    tfl-inspect
    tfl-verify
    tf2tflite
  COMMENT "Generate toolchin configuration"
)

list(APPEND DEPS "${TOOLCHAIN_CONFIG}")

#
# Generate quality test runner
#
set(SOURCE_RUNNER "${CMAKE_CURRENT_SOURCE_DIR}/runner.sh")
set(TARGET_RUNNER "${CMAKE_CURRENT_BINARY_DIR}/runner.sh")

add_custom_command(
  OUTPUT ${TARGET_RUNNER}
  COMMAND ${CMAKE_COMMAND} -E copy "${SOURCE_RUNNER}" "${TARGET_RUNNER}"
  DEPENDS ${SOURCE_RUNNER}
  COMMENT "Generate test runner"
)

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
  COMMENT "Generate rule lib"
)

list(APPEND DEPS "${TARGET_RULE_LIB}")

# Generate dependencies
add_custom_target(tf2tflite_dredd_pb_deps ALL DEPENDS ${DEPS})

add_test(
  NAME tf2tflite_dredd_pb_test
  COMMAND
    "${TARGET_RUNNER}"
    "${TOOLCHAIN_CONFIG}"
    ${KEYS}
)
