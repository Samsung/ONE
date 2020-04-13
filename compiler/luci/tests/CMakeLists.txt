# TODO use local test.recipe files for small networks
file(GLOB RECIPES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*/test.recipe")

foreach(RECIPE IN ITEMS ${RECIPES})
  get_filename_component(RECIPE_PREFIX ${RECIPE} DIRECTORY)

  set(RECIPE_SOURCE_FILE "${RECIPE_PREFIX}.recipe")
  set(RECIPE_SOURCE_TARGET lucitests_${RECIPE_PREFIX}_recipe)

  set(RECIPE_OUTPUT_FILE "${RECIPE_PREFIX}.tflite")
  set(RECIPE_OUTPUT_TARGET lucitests_${RECIPE_PREFIX}_tflite)

  set(CIRCLE_OUTPUT_FILE "${RECIPE_PREFIX}.circle")
  set(CIRCLE_OUTPUT_TARGET lucitests_${RECIPE_PREFIX}_circle)

  # Copy .recipe
  add_custom_target(${RECIPE_SOURCE_TARGET}
                    ALL ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/${RECIPE}"
                                                 "${CMAKE_CURRENT_BINARY_DIR}/${RECIPE_SOURCE_FILE}"
                    COMMENT "Generate ${RECIPE_PREFIX}.recipe")

  # Generate .tflite
  add_custom_target(${RECIPE_OUTPUT_TARGET}
                    ALL $<TARGET_FILE:tflchef-file> ${RECIPE_SOURCE_FILE} ${RECIPE_OUTPUT_FILE}
                    DEPENDS ${RECIPE_SOURCE_TARGET}
                    COMMENT "Generate ${RECIPE_PREFIX}.tflite")

  # Generate .circle
  add_custom_target(${CIRCLE_OUTPUT_TARGET}
                    ALL $<TARGET_FILE:tflite2circle> ${RECIPE_OUTPUT_FILE} ${CIRCLE_OUTPUT_FILE}
                    DEPENDS ${RECIPE_OUTPUT_TARGET}
                    COMMENT "Generate ${RECIPE_PREFIX}.circle")

  list(APPEND TESTS ${RECIPE_PREFIX})
endforeach(RECIPE)

# Generate from res/TensorFlowLiteRecipes
nncc_find_resource(TensorFlowLiteRecipes)
set(TENSORFLOWLITERECIPES_DIR "${TensorFlowLiteRecipes_DIR}")

file(GLOB RECIPES RELATIVE ${TENSORFLOWLITERECIPES_DIR} "${TENSORFLOWLITERECIPES_DIR}/*/test.recipe")

foreach(RECIPE IN ITEMS ${RECIPES})
  get_filename_component(RECIPE_PREFIX ${RECIPE} DIRECTORY)

  set(RECIPE_SOURCE_FILE "${RECIPE_PREFIX}.recipe")
  set(RECIPE_SOURCE_TARGET lucitests_res_${RECIPE_PREFIX}_recipe)

  set(RECIPE_OUTPUT_FILE "${RECIPE_PREFIX}.tflite")
  set(RECIPE_OUTPUT_TARGET lucitests_res_${RECIPE_PREFIX}_tflite)

  set(CIRCLE_OUTPUT_FILE "${RECIPE_PREFIX}.circle")
  set(CIRCLE_OUTPUT_TARGET lucitests_res_${RECIPE_PREFIX}_circle)

  # Copy .recipe
  add_custom_target(${RECIPE_SOURCE_TARGET}
                    ALL ${CMAKE_COMMAND} -E copy "${TENSORFLOWLITERECIPES_DIR}/${RECIPE}"
                                                 "${CMAKE_CURRENT_BINARY_DIR}/${RECIPE_SOURCE_FILE}"
                    COMMENT "Generate ${RECIPE_PREFIX}.recipe")

  # Generate .tflite
  add_custom_target(${RECIPE_OUTPUT_TARGET}
                    ALL $<TARGET_FILE:tflchef-file> ${RECIPE_SOURCE_FILE} ${RECIPE_OUTPUT_FILE}
                    DEPENDS ${RECIPE_SOURCE_TARGET}
                    COMMENT "Generate ${RECIPE_PREFIX}.tflite")

  # Generate .circle
  add_custom_target(${CIRCLE_OUTPUT_TARGET}
                    ALL $<TARGET_FILE:tflite2circle> ${RECIPE_OUTPUT_FILE} ${CIRCLE_OUTPUT_FILE}
                    DEPENDS ${RECIPE_OUTPUT_TARGET}
                    COMMENT "Generate ${RECIPE_PREFIX}.circle")

  list(APPEND TESTS ${RECIPE_PREFIX})
endforeach(RECIPE)

macro(addread NAME)
  list(APPEND DAILY_READ_TESTS ${NAME})
endmacro(addread)

macro(addwrite NAME)
  list(APPEND DAILY_WRITE_TESTS ${NAME})
endmacro(addwrite)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

add_test(NAME luci_unit_readtest
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/readverify.sh"
          "${CMAKE_CURRENT_BINARY_DIR}"
          "$<TARGET_FILE:luci_readtester>"
          ${DAILY_READ_TESTS}
)

add_test(NAME luci_unit_writetest
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/writeverify.sh"
          "${CMAKE_CURRENT_BINARY_DIR}"
          "$<TARGET_FILE:luci_writetester>"
          ${DAILY_WRITE_TESTS}
)
