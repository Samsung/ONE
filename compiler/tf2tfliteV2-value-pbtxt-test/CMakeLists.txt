find_package(PythonInterp 3 QUIET)
find_package(PythonLibs 3 QUIET)

if(NOT ${PYTHONINTERP_FOUND})
  message(STATUS "Build tf2tfliteV2-value-pbtxt-test: FALSE (Python3 is missing)")
  return()
endif()

if(${PYTHON_VERSION_MINOR} LESS 3)
  message(STATUS "Build tf2tfliteV2-value-pbtxt-test: FALSE (You need to install Python version higher than 3.3)")
  return()
endif()

nnas_include(TargetRequire)

unset(REQUIRED_TARGETS)
list(APPEND REQUIRED_TARGETS tfkit)
list(APPEND REQUIRED_TARGETS tf2tfliteV2)
list(APPEND REQUIRED_TARGETS nnkit-run)
list(APPEND REQUIRED_TARGETS nnkit_tf_backend)
list(APPEND REQUIRED_TARGETS nnkit_tflite_backend)
list(APPEND REQUIRED_TARGETS nnkit_randomize_action)
list(APPEND REQUIRED_TARGETS nnkit_HDF5_export_action)
list(APPEND REQUIRED_TARGETS nnkit_HDF5_import_action)
list(APPEND REQUIRED_TARGETS i5diff)
TargetRequire_Return(${REQUIRED_TARGETS})

message(STATUS "tf2tfliteV2-value-pbtxt-test: run tests")

# Create python virtual environment
set(VIRTUALENV "${CMAKE_CURRENT_BINARY_DIR}/venv")

add_custom_command(
  OUTPUT ${VIRTUALENV}
  COMMAND ${PYTHON_EXECUTABLE} -m venv ${VIRTUALENV}
)

# Copy requirements.txt and install required pip packages
set(REQUIREMENTS_FILE "requirements.txt")
set(REQUIREMENTS_SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${REQUIREMENTS_FILE}")
set(REQUIREMENTS_BIN_PATH "${CMAKE_CURRENT_BINARY_DIR}/${REQUIREMENTS_FILE}")

add_custom_command(
  OUTPUT ${REQUIREMENTS_BIN_PATH}
  COMMAND ${CMAKE_COMMAND} -E copy ${REQUIREMENTS_SRC_PATH} ${REQUIREMENTS_BIN_PATH}
  COMMAND ${VIRTUALENV}/bin/python -m pip install --upgrade pip setuptools --timeout 100
  COMMAND ${VIRTUALENV}/bin/python -m pip install -r requirements.txt --upgrade --timeout 100
  DEPENDS ${VIRTUALENV} ${REQUIREMENTS_SRC_PATH}
)

add_custom_target(tf2tfliteV2_value_pbtxt_python_deps ALL
  DEPENDS ${VIRTUALENV} ${REQUIREMENTS_BIN_PATH} #${TF2TFLITEV2_BIN_PATH}
)

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

# Get tf2tfliteV2 binary path
get_target_property(TF2TFLITEV2_BIN_DIR tf2tfliteV2 BINARY_DIR)
set(TF2TFLITEV2_PATH "${TF2TFLITEV2_BIN_DIR}/tf2tfliteV2.py")

add_custom_command(
  OUTPUT ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E remove -f ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'NNKIT_RUN_PATH=\"$<TARGET_FILE:nnkit-run>\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'TF_BACKEND_PATH=\"$<TARGET_FILE:nnkit_tf_backend>\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'TFLITE_BACKEND_PATH=\"$<TARGET_FILE:nnkit_tflite_backend>\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'RANDOMIZE_ACTION_PATH=\"$<TARGET_FILE:nnkit_randomize_action>\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'HDF5_EXPORT_ACTION_PATH=\"$<TARGET_FILE:nnkit_HDF5_export_action>\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'HDF5_IMPORT_ACTION_PATH=\"$<TARGET_FILE:nnkit_HDF5_import_action>\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'I5DIFF_PATH=\"$<TARGET_FILE:i5diff>\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'TF2TFLITEV2_PATH=\"${TF2TFLITEV2_PATH}\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'VIRTUALENV=\"${VIRTUALENV}\"' >> ${TEST_CONFIG}
  DEPENDS
    i5diff
    nnkit-run
    nnkit_tf_backend
    nnkit_tflite_backend
    nnkit_randomize_action
    nnkit_HDF5_export_action
    nnkit_HDF5_import_action
    tf2tfliteV2
    tf2tfliteV2_value_pbtxt_python_deps
  COMMENT "Generate test configuration"
)

list(APPEND TEST_DEPS "${TEST_CONFIG}")

# This "tf2tfliteV2_value_pbtxt_test_deps" target enforces CMake to generate all the dependencies during "build" phase
add_custom_target(tf2tfliteV2_value_pbtxt_test_deps ALL DEPENDS ${TEST_DEPS})

# Run tests
# add_test(
#   NAME tf2tfliteV2_value_pbtxt_test
#   COMMAND "${TEST_RUNNER}"
#           "${TEST_CONFIG}"
#           "${CMAKE_CURRENT_BINARY_DIR}"
#           ${TEST_NAMES}
# )
