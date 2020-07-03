nnas_include(TargetRequire)

unset(REQUIRED_TARGETS)
list(APPEND REQUIRED_TARGETS circlechef)
list(APPEND REQUIRED_TARGETS circle-inspect)
list(APPEND REQUIRED_TARGETS circle-verify)
list(APPEND REQUIRED_TARGETS circle2circle)
list(APPEND REQUIRED_TARGETS dredd_rule_lib)
list(APPEND REQUIRED_TARGETS tflchef)
list(APPEND REQUIRED_TARGETS tflite2circle)
TargetRequire_Return(${REQUIRED_TARGETS})

nncc_find_resource(TensorFlowLiteRecipes)
nncc_find_resource(CircleRecipes)

set(TFLITE_RECIPE_REPO "${TensorFlowLiteRecipes_DIR}")
set(CIRCLE_RECIPE_REPO "${CircleRecipes_DIR}")
unset(RECIPE_REPO)

set(TEST_RECIPE_FILENAME "test.recipe")
set(TEST_RULE_FILENAME "test.rule")

unset(TEST_DEPS)
unset(TEST_NAMES)

set(options "")
set(oneValueArgs "")
set(multiValueArgs PASS)

macro(Add RECIPE)
  if(NOT EXISTS "${TFLITE_RECIPE_REPO}/${RECIPE}/test.recipe")
    if(NOT EXISTS "${CIRCLE_RECIPE_REPO}/${RECIPE}/test.recipe")
      message(FATAL_ERROR "Missing recipe of '${RECIPE}' test")
    else()
      set(RECIPE_REPO ${CIRCLE_RECIPE_REPO})
    endif()
  else()
    set(RECIPE_REPO ${TFLITE_RECIPE_REPO})
  endif()

  if(NOT EXISTS "${RECIPE_REPO}/${RECIPE}/test.rule")
    message(FATAL_ERROR "Missing rule of '${RECIPE}' test")
  endif()

  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  unset(OPT_OPTIONS)
  foreach(src ${ARG_PASS})
    # option = "--${src}"
    list(APPEND OPT_OPTIONS "--${src}")
  endforeach(src ${ARG_PASS})

  set(RECIPE_FILE "${RECIPE}.recipe")
  set(RECIPE_SOURCE_PATH "${RECIPE_REPO}/${RECIPE}/${TEST_RECIPE_FILENAME}")
  set(RECIPE_BINARY_PATH "${CMAKE_CURRENT_BINARY_DIR}/${RECIPE_FILE}")
  
  set(RULE_FILE "${RECIPE}.rule")
  set(RULE_SOURCE_PATH "${RECIPE_REPO}/${RECIPE}/${TEST_RULE_FILENAME}")
  set(RULE_BINARY_PATH "${CMAKE_CURRENT_BINARY_DIR}/${RULE_FILE}")

  set(TFLITE_FILE "${RECIPE}.tflite")
  set(TFLITE_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${TFLITE_FILE}")

  set(CIRCLE_FILE "${RECIPE}.circle")
  set(CIRCLE_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${CIRCLE_FILE}")

  set(OPT_CIRCLE_FILE "${RECIPE}.opt.circle")
  set(OPT_CIRCLE_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${OPT_CIRCLE_FILE}")

  # Copy .recipe
  add_custom_command(OUTPUT ${RECIPE_BINARY_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy "${RECIPE_SOURCE_PATH}" "${RECIPE_BINARY_PATH}"
    DEPENDS ${RECIPE_SOURCE_PATH}
    COMMENT "Generate ${RECIPE_FILE}"
  )

  # Copy .rule
  add_custom_command(OUTPUT ${RULE_BINARY_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy "${RULE_SOURCE_PATH}" "${RULE_BINARY_PATH}"
    DEPENDS ${RULE_SOURCE_PATH}
    COMMENT "Generate ${RULE_FILE}"
  )

  if(${RECIPE_REPO} STREQUAL ${TFLITE_RECIPE_REPO})
    # Generate .tflite
    add_custom_command(OUTPUT ${TFLITE_OUTPUT_PATH}
      COMMAND $<TARGET_FILE:tflchef-file> ${RECIPE_BINARY_PATH} ${TFLITE_OUTPUT_PATH}
      DEPENDS $<TARGET_FILE:tflchef-file> ${RECIPE_BINARY_PATH}
      COMMENT "Generate ${TFLITE_FILE}"
    )

    # Generate .circle
    add_custom_command(OUTPUT ${CIRCLE_OUTPUT_PATH}
      COMMAND $<TARGET_FILE:tflite2circle> ${TFLITE_OUTPUT_PATH} ${CIRCLE_OUTPUT_PATH}
      DEPENDS $<TARGET_FILE:tflite2circle> ${TFLITE_OUTPUT_PATH}
      COMMENT "Generate ${CIRCLE_FILE}"
    )

    list(APPEND TEST_DEPS ${TFLITE_OUTPUT_PATH})
  else()
   # Generate .circle
    add_custom_command(OUTPUT ${CIRCLE_OUTPUT_PATH}
      COMMAND $<TARGET_FILE:circlechef-file> ${RECIPE_BINARY_PATH} ${CIRCLE_OUTPUT_PATH}
      DEPENDS $<TARGET_FILE:circlechef-file> ${RECIPE_BINARY_PATH}
      COMMENT "Generate ${CIRCLE_FILE}"
    )
  endif()

  # Generate optimized .circle
  add_custom_command(OUTPUT ${OPT_CIRCLE_OUTPUT_PATH}
    COMMAND $<TARGET_FILE:circle2circle> ${OPT_OPTIONS} ${CIRCLE_OUTPUT_PATH} ${OPT_CIRCLE_OUTPUT_PATH}
    DEPENDS $<TARGET_FILE:circle2circle> ${CIRCLE_OUTPUT_PATH}
    COMMENT "Generate ${OPT_CIRCLE_FILE}"
  )

  list(APPEND TEST_DEPS ${RECIPE_BINARY_PATH} ${RULE_BINARY_PATH}
                        ${CIRCLE_OUTPUT_PATH} ${OPT_CIRCLE_OUTPUT_PATH})
  list(APPEND TEST_NAMES ${RECIPE})
endmacro(Add)

# Read "test.lst"
include("test.lst")

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
  COMMAND ${CMAKE_COMMAND} -E echo 'CIRCLE_INSPECT_PATH=\"$<TARGET_FILE:circle-inspect>\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'CIRCLE_VERIFY_PATH=\"$<TARGET_FILE:circle-verify>\"' >> ${TEST_CONFIG}
  DEPENDS 
    circle-inspect
    circle-verify
  COMMENT "Generate test configuration"
)

list(APPEND TEST_DEPS "${TEST_CONFIG}")

#
# copy rule-lib.sh (a library of shell script functions)
#

# getting path for rule-lib.sh in dredd-rule-lib
get_target_property(DREDD_RULE_LIB_DIR dredd_rule_lib BINARY_DIR)

set(RULE_LIB_SOURCE_PATH "${DREDD_RULE_LIB_DIR}/rule-lib.sh")
set(RULE_LIB_BINARY_PATH "${CMAKE_CURRENT_BINARY_DIR}/rule-lib.sh")

add_custom_command(
  OUTPUT ${RULE_LIB_BINARY_PATH}
  COMMAND ${CMAKE_COMMAND} -E copy "${RULE_LIB_SOURCE_PATH}" "${RULE_LIB_BINARY_PATH}"
  DEPENDS ${RULE_LIB_SOURCE_PATH}
  COMMENT "Generate rule lib"
)

list(APPEND TEST_DEPS "${RULE_LIB_BINARY_PATH}")

# Generate dependencies
add_custom_target(circle2circle_dredd_recipe_test ALL DEPENDS ${TEST_DEPS})

# Run tests
add_test(
  NAME circle2circle_dredd_recipe_test
  COMMAND "${TEST_RUNNER}"
          "${TEST_CONFIG}"
          "${CMAKE_CURRENT_BINARY_DIR}"
          ${TEST_NAMES}
)
