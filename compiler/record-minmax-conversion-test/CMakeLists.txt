find_package(PythonInterp 3 QUIET)
find_package(PythonLibs 3 QUIET)

if(NOT ${PYTHONINTERP_FOUND})
  message(STATUS "Build record-minmax-conversion-test: FALSE (Python3 is missing)")
  return()
endif()

if(${PYTHON_VERSION_MINOR} LESS 3)
  message(STATUS "Build record-minmax-conversion-test: FALSE (You need to install Python version higher than 3.3)")
  return()
endif()

message(STATUS "record-minmax-conversion-test: run tests")

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
  COMMAND ${VIRTUALENV}/bin/python -m pip --default-timeout=1000 install --upgrade pip setuptools
  COMMAND ${VIRTUALENV}/bin/python -m pip --default-timeout=1000 install -r requirements.txt --upgrade
  DEPENDS ${VIRTUALENV} ${REQUIREMENTS_SRC_PATH}
)

add_custom_target(record-minmax-conversion-test_python_deps ALL
  DEPENDS ${VIRTUALENV} ${REQUIREMENTS_BIN_PATH}
)

unset(RECORD_MINMAX_CONVERSION_TEST)

macro(addTest NAME)
  list(APPEND RECORD_MINMAX_CONVERSION_TEST ${NAME})
endmacro(addTest)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

nncc_find_resource(TensorFlowLiteRecipes)
set(TEST_REPO "${TensorFlowLiteRecipes_DIR}")
set(TEST_RECIPE_FILENAME "test.recipe")

unset(TEST_DEPS)

foreach(TEST IN ITEMS ${RECORD_MINMAX_CONVERSION_TEST})
  set(RECIPE "${TEST}/${TEST_RECIPE_FILENAME}")
  set(RECIPE_SOURCE_FILE "${TEST}.recipe")
  set(RECIPE_OUTPUT_FILE "${TEST}.tflite")
  set(CIRCLE_OUTPUT_FILE "${TEST}.circle")

  # Copy .recipe
  add_custom_command(OUTPUT "${RECIPE_SOURCE_FILE}"
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                             "${TEST_REPO}/${RECIPE}" "${RECIPE_SOURCE_FILE}"
                     DEPENDS "${TEST_REPO}/${RECIPE}"
                     COMMENT "Generating ${RECIPE_SOURCE_FILE}")

  # Generate .tflite
  add_custom_command(OUTPUT "${RECIPE_OUTPUT_FILE}"
                     COMMAND tflchef-file "${RECIPE_SOURCE_FILE}" "${RECIPE_OUTPUT_FILE}"
                     DEPENDS tflchef-file "${RECIPE_SOURCE_FILE}"
                     COMMENT "Generating ${RECIPE_OUTPUT_FILE}")

  # Generate .circle
  add_custom_command(OUTPUT "${CIRCLE_OUTPUT_FILE}"
                     COMMAND tflite2circle "${RECIPE_OUTPUT_FILE}" "${CIRCLE_OUTPUT_FILE}"
                     DEPENDS tflite2circle "${RECIPE_OUTPUT_FILE}"
                     COMMENT "Generating ${CIRCLE_OUTPUT_FILE}")

  list(APPEND TEST_DEPS "${CIRCLE_OUTPUT_FILE}")
endforeach(TEST IN ITEMS ${RECORD_MINMAX_CONVERSION_TEST})

###
### Generate test.config
###
set(TEST_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/test.config")

add_custom_command(
  OUTPUT ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E remove -f ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'RECORD_MINMAX_PATH=\"$<TARGET_FILE:record-minmax>\"' >> ${TEST_CONFIG}
  DEPENDS record-minmax
  COMMENT "Generate test configuration"
)

list(APPEND TEST_DEPS "${TEST_CONFIG}")

# This enforces CMake to generate all the dependencies during "build" phase
add_custom_target(record_minmax_conversion_test_deps ALL DEPENDS ${TEST_DEPS} record-minmax-conversion-test_python_deps)

# Run tests
add_test(
  NAME record_minmax_conversion_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/testall.sh"
          "${TEST_CONFIG}"
          "${CMAKE_CURRENT_BINARY_DIR}"
          ${RECORD_MINMAX_CONVERSION_TEST}
)
