set(eval_verifier_FILE "eval_verifier.py")
set(eval_verifier_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${eval_verifier_FILE}")
set(eval_verifier_BIN "${CMAKE_CURRENT_BINARY_DIR}/${eval_verifier_FILE}")

configure_file(${eval_verifier_SRC} ${eval_verifier_BIN})

add_custom_target(eval_verifier ALL DEPENDS ${eval_verifier_BIN})

install(FILES ${eval_verifier_BIN} DESTINATION bin)

macro(addeval NAME)
  list(APPEND DAILY_EVAL_TESTS ${NAME})
endmacro(addeval)

# Read "test.lst"
include("test.lst")

# Generate tflite and circle from res/TensorFlowLiteRecipes
nncc_find_resource(TensorFlowLiteRecipes)
set(TENSORFLOWLITERECIPES_DIR "${TensorFlowLiteRecipes_DIR}")

foreach(TEST IN ITEMS ${DAILY_EVAL_TESTS})
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
add_custom_target(luci_eval_testfiles ALL DEPENDS ${TESTFILES})

add_subdirectory(tester)

add_test(NAME luci_value_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/evalverify.sh"
          "${CMAKE_CURRENT_BINARY_DIR}"
          "$<TARGET_FILE:luci_interpreter_tester>"
          ${DAILY_EVAL_TESTS}
)
