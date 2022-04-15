nnas_include(TargetRequire)

unset(REQUIRED_TARGETS)
list(APPEND REQUIRED_TARGETS tf2circle)
list(APPEND REQUIRED_TARGETS tfkit)
TargetRequire_Return(${REQUIRED_TARGETS})

message(STATUS "tf2circle-conversion-test: run tests")

nncc_find_resource(TensorFlowTests)

#
# Copy [PREFIX]/test.pbtxt to PREFIX.pbtxt in binary folder
# Copy [PREFIX]/test.info to PREFIX.info in binary folder
# Copy [PREFIX]/customop.conf to PREFIX_customop.conf in binary folder
# Encode PREFIX.pbtxt to PREFIX.pb
#
set(TEST_REPO "${TensorFlowTests_DIR}")
set(TEST_PBTXT_FILENAME "test.pbtxt")
set(TEST_INFO_FILENAME "test.info")
set(TEST_CUSTOMOP_CONF_FILENAME "customop.conf")

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
  set(CUSTOMOP_CONF_SOURCE_PATH "${TEST_REPO}/${PREFIX}/${TEST_CUSTOMOP_CONF_FILENAME}")

  set(PBTXT_FILE "${PREFIX}.pbtxt")
  set(PBTXT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${PBTXT_FILE}")

  set(INFO_FILE "${PREFIX}.info")
  set(INFO_PATH "${CMAKE_CURRENT_BINARY_DIR}/${INFO_FILE}")

  set(CUSTOMOP_CONF_FILE "${PREFIX}.${TEST_CUSTOMOP_CONF_FILENAME}") # ex) CustomOp_001.customop.conf
  set(CUSTOMOP_CONF_PATH "${CMAKE_CURRENT_BINARY_DIR}/${CUSTOMOP_CONF_FILE}")

  set(PB_FILE "${PREFIX}.pb")
  set(PB_PATH "${CMAKE_CURRENT_BINARY_DIR}/${PB_FILE}")

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

  # Generate .pb from .pbtxt
  add_custom_command(OUTPUT ${PB_PATH}
    COMMAND $<TARGET_FILE:tfkit> encode ${PBTXT_PATH} ${PB_PATH}
    DEPENDS ${PBTXT_PATH}
    COMMENT "Generate ${PB_FILE}"
  )

  list(APPEND TEST_DEPS ${INFO_PATH} ${PB_PATH})

  if (EXISTS "${CUSTOMOP_CONF_SOURCE_PATH}")

    # Copy customop.conf
    add_custom_command(OUTPUT ${CUSTOMOP_CONF_PATH}
      COMMAND ${CMAKE_COMMAND} -E copy "${CUSTOMOP_CONF_SOURCE_PATH}" "${CUSTOMOP_CONF_PATH}"
      DEPENDS ${CUSTOMOP_CONF_SOURCE_PATH}
      COMMENT "Generate ${CUSTOMOP_CONF_FILE}"
    )

    list(APPEND TEST_DEPS ${CUSTOMOP_CONF_PATH})

  endif (EXISTS "${CUSTOMOP_CONF_SOURCE_PATH}")

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

###
### Generate test.config
###
set(TEST_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/test.config")

add_custom_command(
  OUTPUT ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E remove -f ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'TF2CIRCLE_PATH=\"$<TARGET_FILE:tf2circle>\"' >> ${TEST_CONFIG}
  DEPENDS
    tf2circle
  COMMENT "Generate test configuration"
)

list(APPEND TEST_DEPS "${TEST_CONFIG}")

# This "tf2circle_conversion_test_deps" target enforces CMake to generate all the dependencies during "build" phase
add_custom_target(tf2circle_conversion_test_deps ALL DEPENDS ${TEST_DEPS})

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Run tests
add_test(
  NAME tf2circle_conversion_test
  COMMAND "${TEST_RUNNER}"
          "${TEST_CONFIG}"
          "${CMAKE_CURRENT_BINARY_DIR}"
          ${TEST_NAMES}
)
