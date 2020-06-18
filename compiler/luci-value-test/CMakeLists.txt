find_package(PythonInterp 3 QUIET)
find_package(PythonLibs 3 QUIET)

if(NOT ${PYTHONINTERP_FOUND})
  message(STATUS "Build luci-value-test: FALSE (Python3 is missing)")
  return()
endif()

if(${PYTHON_VERSION_MINOR} LESS 3)
  message(STATUS "Build luci-value-test: FALSE (You need to install Python version higher than 3.3)")
  return()
endif()

message(STATUS "luci-value-test: run tests")

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

add_custom_target(luci-value-test_python_deps ALL
  DEPENDS ${VIRTUALENV} ${REQUIREMENTS_BIN_PATH}
)

unset(LUCI_VALUE_TESTS)

macro(addeval NAME)
  list(APPEND LUCI_VALUE_TESTS ${NAME})
endmacro(addeval)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

# Generate tflite and circle from res/TensorFlowLiteRecipes
nncc_find_resource(TensorFlowLiteRecipes)
set(TENSORFLOWLITERECIPES_DIR "${TensorFlowLiteRecipes_DIR}")

unset(TESTFILES)

foreach(TEST IN ITEMS ${LUCI_VALUE_TESTS})
  set(RECIPE "${TEST}/test.recipe")
  get_filename_component(RECIPE_PREFIX ${RECIPE} DIRECTORY)

  set(RECIPE_SOURCE_FILE "${RECIPE_PREFIX}.recipe")
  set(RECIPE_OUTPUT_FILE "${RECIPE_PREFIX}.tflite")
  set(CIRCLE_OUTPUT_FILE "${RECIPE_PREFIX}.circle")

  # Copy .recipe
  add_custom_command(OUTPUT "${RECIPE_SOURCE_FILE}"
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                             "${TENSORFLOWLITERECIPES_DIR}/${RECIPE}" "${RECIPE_SOURCE_FILE}"
                     DEPENDS "${TENSORFLOWLITERECIPES_DIR}/${RECIPE}"
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

  list(APPEND TESTFILES "${CIRCLE_OUTPUT_FILE}")
endforeach(TEST)

# Generate dependencies
add_custom_target(luci_eval_testfiles ALL DEPENDS ${TESTFILES} luci-value-test_python_deps)

add_subdirectory(tester)

add_test(NAME luci_value_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/evalverify.sh"
          "${CMAKE_CURRENT_BINARY_DIR}"
          ${LUCI_VALUE_TESTS}
)
