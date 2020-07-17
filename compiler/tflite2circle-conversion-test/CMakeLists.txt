nnas_include(TargetRequire)

unset(REQUIRED_TARGETS)
list(APPEND REQUIRED_TARGETS tflite2circle)
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

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

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
          "${ARTIFACTS_BIN_PATH}"
          ${TEST_NAMES}
)
