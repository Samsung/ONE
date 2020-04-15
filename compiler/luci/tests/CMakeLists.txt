# TODO use local test.recipe files for small networks
file(GLOB RECIPES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*/test.recipe")

foreach(RECIPE IN ITEMS ${RECIPES})
  get_filename_component(RECIPE_PREFIX ${RECIPE} DIRECTORY)

  set(RECIPE_SOURCE_FILE "${RECIPE_PREFIX}.recipe")
  set(RECIPE_OUTPUT_FILE "${RECIPE_PREFIX}.tflite")
  set(CIRCLE_OUTPUT_FILE "${RECIPE_PREFIX}.circle")

  # Copy .recipe
  add_custom_command(OUTPUT "${RECIPE_SOURCE_FILE}"
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                             "${CMAKE_CURRENT_SOURCE_DIR}/${RECIPE}" "${RECIPE_SOURCE_FILE}"
                     DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${RECIPE}"
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
endforeach(RECIPE)

# Generate from res/TensorFlowLiteRecipes
nncc_find_resource(TensorFlowLiteRecipes)
set(TENSORFLOWLITERECIPES_DIR "${TensorFlowLiteRecipes_DIR}")

macro(testcase NAME)
  list(APPEND TESTNAMES ${NAME})
endmacro(addname)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

foreach(TESTNAME IN ITEMS ${TESTNAMES})
  set(RECIPE "${TESTNAME}/test.recipe")
  set(RECIPE_SOURCE_FILE "${TESTNAME}.recipe")
  set(RECIPE_OUTPUT_FILE "${TESTNAME}.tflite")
  set(CIRCLE_OUTPUT_FILE "${TESTNAME}.circle")

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
endforeach(RECIPE)

# Add a dummy target to create a target-level dependency.
# TODO Find a way to create dependency between CTest tests (added below) and generated testfiles.
add_custom_target(luci_testfiles ALL DEPENDS ${TESTFILES})

add_test(NAME luci_unit_readtest
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/readverify.sh"
          "${CMAKE_CURRENT_BINARY_DIR}"
          "$<TARGET_FILE:luci_readtester>"
          ${TESTNAMES}
)

add_test(NAME luci_unit_writetest
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/writeverify.sh"
          "${CMAKE_CURRENT_BINARY_DIR}"
          "$<TARGET_FILE:luci_writetester>"
          ${TESTNAMES}
)
