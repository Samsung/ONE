nnas_include(TargetRequire)

unset(REQUIRED_TARGETS)
list(APPEND REQUIRED_TARGETS tf2circle)
list(APPEND REQUIRED_TARGETS tfkit)
list(APPEND REQUIRED_TARGETS nnkit-run)
list(APPEND REQUIRED_TARGETS nnkit_tf_backend)
list(APPEND REQUIRED_TARGETS nnkit_randomize_action)
list(APPEND REQUIRED_TARGETS nnkit_HDF5_export_action)
list(APPEND REQUIRED_TARGETS nnkit_HDF5_import_action)
TargetRequire_Return(${REQUIRED_TARGETS})

message(STATUS "tf2circle-value_pbtxt-remote-test: run tests")

nncc_find_resource(TensorFlowTests)

#
# Copy [PREFIX]/test.pbtxt to PREFIX.pbtxt in binary folder
# Copy [PREFIX]/test.info to PREFIX.info in binary folder
# Copy [PREFIX]/customop.conf to PREFIX.customop.conf in binary folder
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

# Read "test.lst" if exists
include("test.lst" OPTIONAL)

# Do not make test if there are no remote machine information
if(NOT REMOTE_IP)
  return()
endif(NOT REMOTE_IP)

if(NOT REMOTE_USER)
  return()
endif(NOT REMOTE_USER)

# Do not make test if there are no testcases
if(NOT TESTCASES)
  return()
endif(NOT TESTCASES)

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
  COMMAND ${CMAKE_COMMAND} -E echo 'NNKIT_RUN_PATH=\"$<TARGET_FILE:nnkit-run>\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'TF_BACKEND_PATH=\"$<TARGET_FILE:nnkit_tf_backend>\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'TF2CIRCLE_PATH=\"$<TARGET_FILE:tf2circle>\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'RANDOMIZE_ACTION_PATH=\"$<TARGET_FILE:nnkit_randomize_action>\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'HDF5_EXPORT_ACTION_PATH=\"$<TARGET_FILE:nnkit_HDF5_export_action>\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'HDF5_IMPORT_ACTION_PATH=\"$<TARGET_FILE:nnkit_HDF5_import_action>\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'MODEL2NNPKG_PATH=\"${NNAS_PROJECT_SOURCE_DIR}/tools/nnpackage_tool/model2nnpkg/model2nnpkg.sh\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'NNPKG_TEST_PATH=\"${NNAS_PROJECT_SOURCE_DIR}/tests/scripts/nnpkg_test.sh\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'RUNTIME_LIBRARY_PATH=\"${NNAS_PROJECT_SOURCE_DIR}/Product/out/\"' >> ${TEST_CONFIG}
  DEPENDS
    nnkit-run
    nnkit_tf_backend
    tf2circle
    nnkit_randomize_action
    nnkit_HDF5_export_action
    nnkit_HDF5_import_action
  COMMENT "Generate test configuration"
)

list(APPEND TEST_DEPS "${TEST_CONFIG}")

# This "tf2circle_value_pbtxt_remote_test_deps" target enforces CMake to generate all the dependencies during "build" phase
add_custom_target(tf2circle_value_pbtxt_remote_test_deps ALL DEPENDS ${TEST_DEPS})

# Run tests
add_test(
  NAME tf2circle_value_pbtxt_remote_test
  COMMAND "${TEST_RUNNER}"
          "${TEST_CONFIG}"
          "${CMAKE_CURRENT_BINARY_DIR}"
          "${REMOTE_IP}"
          "${REMOTE_USER}"
          ${TEST_NAMES}
)
