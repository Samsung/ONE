nnas_include(TargetRequire)

unset(REQUIRED_TARGETS)
list(APPEND REQUIRED_TARGETS tflite2circle)
list(APPEND REQUIRED_TARGETS tflchef)
TargetRequire_Return(${REQUIRED_TARGETS})

nncc_find_resource(TensorFlowLiteRecipes)

set(TEST_REPO "${TensorFlowLiteRecipes_DIR}")
set(TEST_RECIPE_FILENAME "test.recipe")

file(GLOB RECIPES RELATIVE ${TEST_REPO} "${TEST_REPO}/*/${TEST_RECIPE_FILENAME}")

unset(TEST_DEPS)
unset(TEST_NAMES)

foreach(RECIPE IN ITEMS ${RECIPES})
  get_filename_component(PREFIX ${RECIPE} DIRECTORY)
  if(NOT IS_DIRECTORY "${TEST_REPO}/${RECIPE_PREFIX}")
    message(FATAL_ERROR "Missing '${RECIPE_PREFIX}' test")
  endif()

  set(RECIPE_SOURCE_PATH "${TEST_REPO}/${PREFIX}/${TEST_RECIPE_FILENAME}")
  set(RECIPE_FILE "${PREFIX}.recipe")
  set(RECIPE_BINARY_PATH "${CMAKE_CURRENT_BINARY_DIR}/${RECIPE_FILE}")

  set(TFLITE_FILE "${PREFIX}.tflite")
  set(TFLITE_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${TFLITE_FILE}")

  # Copy .recipe
  add_custom_command(OUTPUT ${RECIPE_BINARY_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy "${RECIPE_SOURCE_PATH}" "${RECIPE_BINARY_PATH}"
    DEPENDS ${RECIPE_SOURCE_PATH}
    COMMENT "Generate ${RECIPE_FILE}"
  )

  # Generate .tflite
  add_custom_command(OUTPUT ${TFLITE_OUTPUT_PATH}
    COMMAND $<TARGET_FILE:tflchef-file> ${RECIPE_BINARY_PATH} ${TFLITE_OUTPUT_PATH}
    DEPENDS $<TARGET_FILE:tflchef-file> ${RECIPE_BINARY_PATH}
    COMMENT "Generate ${TFLITE_FILE}"
  )

  list(APPEND TEST_DEPS ${RECIPE_BINARY_PATH} ${TFLITE_OUTPUT_PATH})
  list(APPEND TEST_NAMES ${PREFIX})
endforeach(RECIPE IN ITEMS ${RECIPES})

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
  COMMAND ${CMAKE_COMMAND} -E echo 'TFLITE2CIRCLE_PATH=\"$<TARGET_FILE:tflite2circle>\"' >> ${TEST_CONFIG}
  DEPENDS tflite2circle
  COMMENT "Generate test configuration"
)

list(APPEND TEST_DEPS "${TEST_CONFIG}")

# This "tflite2circle_conversion_test_deps" target enforces CMake to generate all the dependencies during "build" phase
add_custom_target(tflite2circle_conversion_test_deps ALL DEPENDS ${TEST_DEPS})

# Run tests
add_test(
  NAME tflite2circle_conversion_test
  COMMAND "${TEST_RUNNER}"
          "${TEST_CONFIG}"
          "${CMAKE_CURRENT_BINARY_DIR}"
          ${TEST_NAMES}
)
