set(CIRCLECHEF_FILE_PATH $<TARGET_FILE:circlechef-file>)
set(TFLCHEF_FILE_PATH $<TARGET_FILE:tflchef-file>)
set(TFLITE2CIRCLE_PATH $<TARGET_FILE:tflite2circle>)
if(DEFINED ENV{BUILD_HOST_EXEC})
  # TODO use better way to represent path for host executable
  set(CIRCLECHEF_FILE_PATH $ENV{BUILD_HOST_EXEC}/compiler/circlechef/tools/file/circlechef-file)
  set(TFLCHEF_FILE_PATH $ENV{BUILD_HOST_EXEC}/compiler/tflchef/tools/file/tflchef-file)
  set(TFLITE2CIRCLE_PATH $ENV{BUILD_HOST_EXEC}/compiler/tflite2circle/tflite2circle)
  message(STATUS "TFLITE2CIRCLE_PATH = ${TFLITE2CIRCLE_PATH}")
endif(DEFINED ENV{BUILD_HOST_EXEC})

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
                     COMMAND ${TFLCHEF_FILE_PATH} "${RECIPE_SOURCE_FILE}" "${RECIPE_OUTPUT_FILE}"
                     DEPENDS ${TFLCHEF_FILE_PATH} "${RECIPE_SOURCE_FILE}"
                     COMMENT "Generating ${RECIPE_OUTPUT_FILE}")

  # Generate .circle
  add_custom_command(OUTPUT "${CIRCLE_OUTPUT_FILE}"
                     COMMAND ${TFLITE2CIRCLE_PATH} "${RECIPE_OUTPUT_FILE}" "${CIRCLE_OUTPUT_FILE}"
                     DEPENDS ${TFLITE2CIRCLE_PATH} "${RECIPE_OUTPUT_FILE}"
                     COMMENT "Generating ${CIRCLE_OUTPUT_FILE}")

  list(APPEND TESTFILES "${CIRCLE_OUTPUT_FILE}")
endforeach(RECIPE)

# Generate from res/TensorFlowLiteRecipes
nncc_find_resource(TensorFlowLiteRecipes)
set(TENSORFLOWLITERECIPES_DIR "${TensorFlowLiteRecipes_DIR}")

file(GLOB RECIPES RELATIVE ${TENSORFLOWLITERECIPES_DIR} "${TENSORFLOWLITERECIPES_DIR}/*/test.recipe")

foreach(RECIPE IN ITEMS ${RECIPES})
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
                     COMMAND ${TFLCHEF_FILE_PATH} "${RECIPE_SOURCE_FILE}" "${RECIPE_OUTPUT_FILE}"
                     DEPENDS ${TFLCHEF_FILE_PATH} "${RECIPE_SOURCE_FILE}"
                     COMMENT "Generating ${RECIPE_OUTPUT_FILE}")

  # Generate .circle
  add_custom_command(OUTPUT "${CIRCLE_OUTPUT_FILE}"
                     COMMAND ${TFLITE2CIRCLE_PATH} "${RECIPE_OUTPUT_FILE}" "${CIRCLE_OUTPUT_FILE}"
                     DEPENDS ${TFLITE2CIRCLE_PATH} "${RECIPE_OUTPUT_FILE}"
                     COMMENT "Generating ${CIRCLE_OUTPUT_FILE}")

  list(APPEND TESTFILES "${CIRCLE_OUTPUT_FILE}")
endforeach(RECIPE)

# Generate from res/CircleRecipes
# NOTE duplicate names should not exist or test may be incorrect
nncc_find_resource(CircleRecipes)
set(CIRCLERECIPES_DIR "${CircleRecipes_DIR}")

file(GLOB RECIPES2 RELATIVE ${CIRCLERECIPES_DIR} "${CIRCLERECIPES_DIR}/*/test.recipe")

foreach(RECIPE IN ITEMS ${RECIPES2})
  get_filename_component(RECIPE_PREFIX ${RECIPE} DIRECTORY)

  set(RECIPE_SOURCE_FILE "${RECIPE_PREFIX}.recipe")
  set(CIRCLE_OUTPUT_FILE "${RECIPE_PREFIX}.circle")

  # Copy .recipe
  add_custom_command(OUTPUT "${RECIPE_SOURCE_FILE}"
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                             "${CIRCLERECIPES_DIR}/${RECIPE}" "${RECIPE_SOURCE_FILE}"
                     DEPENDS "${CIRCLERECIPES_DIR}/${RECIPE}"
                     COMMENT "Generating ${RECIPE_SOURCE_FILE}")

  # Generate .circle
  add_custom_command(OUTPUT "${CIRCLE_OUTPUT_FILE}"
                     COMMAND ${CIRCLECHEF_FILE_PATH} "${RECIPE_SOURCE_FILE}" "${CIRCLE_OUTPUT_FILE}"
                     DEPENDS ${CIRCLECHEF_FILE_PATH} "${RECIPE_SOURCE_FILE}"
                     COMMENT "Generating ${CIRCLE_OUTPUT_FILE}")

  list(APPEND TESTFILES "${CIRCLE_OUTPUT_FILE}")
endforeach(RECIPE)

# Add a dummy target to create a target-level dependency.
# TODO Find a way to create dependency between CTest tests (added below) and generated testfiles.
add_custom_target(luci_testfiles ALL DEPENDS ${TESTFILES})

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

# NOTE $<TARGET_FILE:luci_readtester> is used as-is as test itself should
#      run in target device for cross build also
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
