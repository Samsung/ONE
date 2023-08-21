if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nncc_find_resource(TensorFlowTests)

#
# Copy [PREFIX]/test.pbtxt to PREFIX.pbtxt in binary folder
# Copy [PREFIX]/test.info to PREFIX.info in binary folder
# Encode PREFIX.pbtxt to PREFIX.pb
#
set(TEST_REPO "${TensorFlowTests_DIR}")
set(TEST_PBTXT_FILENAME "test.pbtxt")
set(TEST_INFO_FILENAME "test.info")

unset(TESTCASES)

macro(add NAME)
  list(APPEND TESTCASES ${NAME})
endmacro(add)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

unset(TEST_DEPS)
unset(TEST_NAMES)

foreach(PREFIX IN ITEMS ${TESTCASES})
  if(NOT IS_DIRECTORY "${TEST_REPO}/${PREFIX}")
    message(FATAL_ERROR "Missing '${PREFIX}' test")
  endif()

  set(PBTXT_SOURCE_PATH "${TEST_REPO}/${PREFIX}/${TEST_PBTXT_FILENAME}")
  set(INFO_SOURCE_PATH "${TEST_REPO}/${PREFIX}/${TEST_INFO_FILENAME}")

  set(PBTXT_FILE "${PREFIX}.pbtxt")
  set(PBTXT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${PBTXT_FILE}")

  set(INFO_FILE "${PREFIX}.info")
  set(INFO_PATH "${CMAKE_CURRENT_BINARY_DIR}/${INFO_FILE}")

  # Copy .pbtxt
  add_custom_command(OUTPUT ${PBTXT_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy "${PBTXT_SOURCE_PATH}" "${PBTXT_PATH}"
    DEPENDS ${PBTXT_SOURCE_PATH}
    COMMENT "Generate ${PBTXT_FILE}"
  )

  # Copy .info
  add_custom_command(OUTPUT ${INFO_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy "${INFO_SOURCE_PATH}" "${INFO_PATH}"
    DEPENDS ${INFO_SOURCE_PATH}
    COMMENT "Generate ${INFO_FILE}"
  )

  list(APPEND TEST_DEPS ${INFO_PATH} ${PBTXT_PATH})
  list(APPEND TEST_NAMES ${PREFIX})
endforeach(PREFIX)

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

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

set(VIRTUALENV "${NNCC_OVERLAY_DIR}/venv_2_12_1")

###
### Generate test.config
###
set(TEST_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/test.config")

# Get tf2tfliteV2 binary path
get_target_property(TF2TFLITEV2_BIN_DIR tf2tfliteV2 BINARY_DIR)
set(TF2TFLITEV2_PATH "${TF2TFLITEV2_BIN_DIR}/tf2tfliteV2.py")

add_custom_command(
  OUTPUT ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E remove -f ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'TF2TFLITEV2_PATH=\"${TF2TFLITEV2_PATH}\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'VIRTUALENV=\"${VIRTUALENV}\"' >> ${TEST_CONFIG}
  DEPENDS
    tf2tfliteV2
  COMMENT "Generate test configuration"
)

list(APPEND TEST_DEPS "${TEST_CONFIG}")

# This "tf2tfliteV2_conversion_test_deps" target enforces CMake to generate all the dependencies during "build" phase
add_custom_target(tf2tfliteV2_conversion_test_deps ALL DEPENDS ${TEST_DEPS})

# TODO This test takes a long time and will only be tested once a day.
# Run tests
# add_test(
#   NAME tf2tfliteV2_conversion_test
#   COMMAND "${TEST_RUNNER}"
#           "${TEST_CONFIG}"
#           "${CMAKE_CURRENT_BINARY_DIR}"
#           ${TEST_NAMES}
# )
