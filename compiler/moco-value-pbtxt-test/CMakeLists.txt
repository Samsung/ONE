option(MOCO_VALUE_PBTXT_TEST "Enable moco value test for pbtxt input model" ON)

if(NOT MOCO_VALUE_PBTXT_TEST)
  return()
endif(NOT MOCO_VALUE_PBTXT_TEST)

if(NOT TARGET tfkit)
  message(STATUS "moco: Skip test material preparation as tfkit is not defined")
  return()
endif(NOT TARGET tfkit)

nncc_find_resource(TensorFlowTests)

#
# Copy [Testcase]/test.pbtxt to Testcase.pbtxt in binary folder
# Copy [Testcase]/test.info to Testcase.info in binary folder
# Encode Testcase.pbtxt to Testcase.pb
#
set(TEST_PBTXT_FILE "test.pbtxt")
set(TEST_INFO_FILE "test.info")
set(TEST_REPO "${TensorFlowTests_DIR}") # Where to find tests
set(TEST_SPACE "${CMAKE_CURRENT_BINARY_DIR}") # Where to run tests

unset(TESTCASES)

macro(add NAME)
  list(APPEND TESTCASES ${NAME})
endmacro(add)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

unset(MOCO_VALUE_PBTXT_DEPS)

foreach(PREFIX IN ITEMS ${TESTCASES})
  set(PBTXTFILE "${PREFIX}/${TEST_PBTXT_FILE}")
  get_filename_component(DIR_NAME ${PBTXTFILE} DIRECTORY)

  set(PBTXT_SOURCE_FILE "${DIR_NAME}.pbtxt")
  set(PBTXT_SOURCE_PATH "${TEST_SPACE}/${DIR_NAME}.pbtxt")

  set(PBTXT_INFO_FILE "${DIR_NAME}.info")
  set(PBTXT_INFO_PATH "${TEST_SPACE}/${PBTXT_INFO_FILE}")

  set(PB_OUTPUT_FILE "${DIR_NAME}.pb")
  set(PB_PATH "${TEST_SPACE}/${PB_OUTPUT_FILE}")

  # Copy files
  add_custom_command(
    OUTPUT ${PBTXT_SOURCE_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy "${TEST_REPO}/${DIR_NAME}/${TEST_PBTXT_FILE}"
                                     "${PBTXT_SOURCE_PATH}"
    DEPENDS "${TEST_REPO}/${DIR_NAME}/${TEST_PBTXT_FILE}"
    COMMENT "Copy ${PBTXT_SOURCE_FILE}"
  )

  add_custom_command(
    OUTPUT ${PBTXT_INFO_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy "${TEST_REPO}/${DIR_NAME}/${TEST_INFO_FILE}"
                                     "${PBTXT_INFO_PATH}"
    DEPENDS "${TEST_REPO}/${DIR_NAME}/${TEST_INFO_FILE}"
    COMMENT "Copy ${PBTXT_INFO_FILE}"
  )

  # Use tfkit to encode
  add_custom_command(
    OUTPUT ${PB_OUTPUT_FILE}
    COMMAND $<TARGET_FILE:tfkit> encode ${PBTXT_SOURCE_PATH} ${PB_OUTPUT_FILE}
    DEPENDS tfkit ${PBTXT_SOURCE_PATH}
    COMMENT "Generate ${PB_OUTPUT_FILE}"
  )

  list(APPEND MOCO_VALUE_PBTXT_TESTS ${DIR_NAME})
  list(APPEND MOCO_VALUE_PBTXT_DEPS ${PBTXT_INFO_FILE} ${PB_OUTPUT_FILE})

endforeach(PREFIX)

nnas_find_package(TensorFlow EXACT 1.13 QUIET)
if(NOT TensorFlow_FOUND)
  message(STATUS "moco: Skip adding test as TensorFlow is not found")
  return()
endif(NOT TensorFlow_FOUND)

##
## Copy runall.sh
##
set(TEST_RUNNER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/runall.sh")
set(TEST_RUNNER "${CMAKE_CURRENT_BINARY_DIR}/run-tests")

add_custom_command(
  OUTPUT ${TEST_RUNNER}
  COMMAND ${CMAKE_COMMAND} -E copy "${TEST_RUNNER_SOURCE}" "${TEST_RUNNER}"
  DEPENDS ${TEST_RUNNER_SOURCE}
  COMMENT "Generate test runner"
)

list(APPEND MOCO_VALUE_PBTXT_DEPS "${TEST_RUNNER}")

###
### Generate test.config
###
set(TOOLCHIAN_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/toolchain.config")

add_custom_command(
  OUTPUT ${TOOLCHIAN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E remove -f ${TOOLCHIAN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'NNKIT_RUN_PATH=\"$<TARGET_FILE:nnkit-run>\"' >> ${TOOLCHIAN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'TF_BACKEND_PATH=\"$<TARGET_FILE:nnkit_tf_backend>\"' >> ${TOOLCHIAN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'MOCO_TF_BACKEND_PATH=\"$<TARGET_FILE:nnkit_moco_tf_backend>\"' >> ${TOOLCHIAN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'TF2TFLITE_PATH=\"$<TARGET_FILE:tf2tflite>\"' >> ${TOOLCHIAN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'RANDOMIZE_ACTION_PATH=\"$<TARGET_FILE:nnkit_randomize_action>\"' >> ${TOOLCHIAN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'HDF5_EXPORT_ACTION_PATH=\"$<TARGET_FILE:nnkit_HDF5_export_action>\"' >> ${TOOLCHIAN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'HDF5_IMPORT_ACTION_PATH=\"$<TARGET_FILE:nnkit_HDF5_import_action>\"' >> ${TOOLCHIAN_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'I5DIFF_PATH=\"$<TARGET_FILE:i5diff>\"' >> ${TOOLCHIAN_CONFIG}
  DEPENDS
    nnkit-run
    nnkit_tf_backend
    nnkit_moco_tf_backend
    tf2tflite
    nnkit_randomize_action
    nnkit_HDF5_export_action
    nnkit_HDF5_import_action
    i5diff
  COMMENT "Generate test configuration"
)

list(APPEND MOCO_VALUE_PBTXT_DEPS "${TOOLCHIAN_CONFIG}")

# This target enforces CMake to generate all the dependencies during "build" phase
add_custom_target(moco_value_pbtxt_test_deps ALL DEPENDS ${MOCO_VALUE_PBTXT_DEPS})

# Run tests
add_test(NAME moco_value_pbtxt_test
         COMMAND "${TEST_RUNNER}" "${TOOLCHIAN_CONFIG}" "${TEST_SPACE}" ${MOCO_VALUE_PBTXT_TESTS})
