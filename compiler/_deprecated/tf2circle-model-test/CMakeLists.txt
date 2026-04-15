nnas_include(TargetRequire)

unset(REQUIRED_TARGETS)
list(APPEND REQUIRED_TARGETS tf2circle)
list(APPEND REQUIRED_TARGETS tfkit)
TargetRequire_Return(${REQUIRED_TARGETS})

unset(KEYS)
unset(DEPS)

###
### Add "Contrib" tests
###
macro(Add PREFIX)
  # Let's use CONTRIB prefix to avoid name conflicts with official models
  set(TEST_KEY "CONTRIB.${PREFIX}")

  set(PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/contrib/${PREFIX}")

  set(MODEL_DOWNLOAD_SCRIPT "${PACKAGE_DIR}/model.download")
  set(MODEL_PB_FILE "${PACKAGE_DIR}/model.pb")
  set(MODEL_INFO_FILE "${PACKAGE_DIR}/model.info")
  set(MODEL_MD5SUM_FILE "${PACKAGE_DIR}/model.md5sum")

  # Try to download a model if it is missing
  if(NOT EXISTS "${MODEL_PB_FILE}")
    # TODO Extract this routine as a helper function
    if(NOT EXISTS "${MODEL_DOWNLOAD_SCRIPT}")
      message(FATAL_ERROR "${TEST_KEY} - Download script is missing")
    endif(NOT EXISTS "${MODEL_DOWNLOAD_SCRIPT}")

    execute_process(
      COMMAND ${CMAKE_COMMAND} -D OUTPUT_PATH=${MODEL_PB_FILE} -P "${MODEL_DOWNLOAD_SCRIPT}"
      RESULT_VARIABLE EXITCODE
    )

    if(NOT EXITCODE EQUAL 0)
      message(FATAL_ERROR "${TEST_KEY} - Download fails")
    endif(NOT EXITCODE EQUAL 0)
  endif()

  if(EXISTS "${MODEL_MD5SUM_FILE}")
    # TODO Extract this routine as a helper function
    file(STRINGS "${MODEL_MD5SUM_FILE}" EXPECTED_MD5SUM)
    file(MD5 "${MODEL_PB_FILE}" OBTAINED_MD5SUM)

    if(NOT "${EXPECTED_MD5SUM}" STREQUAL "${OBTAINED_MD5SUM}")
      message(FATAL_ERROR "${TEST_KEY} - Checksum mismatches")
    endif()
  endif()

  # Generate .test file which declares MODEL_PB_PATH and MODEL_INFO_PATH
  set(TEST_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TEST_KEY}.test")

  add_custom_command(
    OUTPUT ${TEST_CONFIG_FILE}
    COMMAND ${CMAKE_COMMAND} -E remove -f ${TEST_CONFIG_FILE}
    COMMAND ${CMAKE_COMMAND} -E echo 'MODEL_PB_PATH="${MODEL_PB_FILE}"' >> ${TEST_CONFIG_FILE}
    COMMAND ${CMAKE_COMMAND} -E echo 'MODEL_INFO_PATH="${MODEL_INFO_FILE}"' >> ${TEST_CONFIG_FILE}
    COMMENT "Generate ${TEST_KEY} configuration"
  )

  list(APPEND KEYS "${TEST_KEY}")
  list(APPEND DEPS "${TEST_CONFIG_FILE}")
endmacro(Add)

include(contrib.lst OPTIONAL)

###
### Generate toolchain.config
###
set(TOOLCHAIN_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/toolchain.config")

add_custom_command(
  OUTPUT ${TOOLCHAIN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E remove -f ${TOOLCHAIN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'TF2CIRCLE_PATH=\"$<TARGET_FILE:tf2circle>\"' >> ${TOOLCHAIN_CONFIG}
  DEPENDS
    tf2circle
  COMMENT "Generate toolchin configuration"
)

list(APPEND DEPS "${TOOLCHAIN_CONFIG}")

##
## Generate test runner
##
set(TEST_RUNNER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/runner.sh")
set(TEST_RUNNER_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/run")

add_custom_command(
  OUTPUT ${TEST_RUNNER_SCRIPT}
  COMMAND ${CMAKE_COMMAND} -E copy "${TEST_RUNNER_SOURCE}" "${TEST_RUNNER_SCRIPT}"
  DEPENDS ${TEST_RUNNER_SOURCE}
  COMMENT "Generate test runner"
)

list(APPEND DEPS "${TEST_RUNNER_SCRIPT}")

### Generate dependencies
add_custom_target(tf2circle_model_test_deps ALL DEPENDS ${DEPS})

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# NOTE This target is not built by default
add_test(
  NAME tf2circle_model_test
  COMMAND
    "${TEST_RUNNER_SCRIPT}"
    "${TOOLCHAIN_CONFIG}"
    ${KEYS}
)
