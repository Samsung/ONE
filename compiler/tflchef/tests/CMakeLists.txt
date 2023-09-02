set(TFLCHEF_FILE_PATH $<TARGET_FILE:tflchef-file>)
set(TFLCHEF_REVERSE_PATH $<TARGET_FILE:tflchef-reverse>)
if(DEFINED ENV{BUILD_HOST_EXEC})
  # TODO use better way to represent path for host executable
  set(TFLCHEF_FILE_PATH $ENV{BUILD_HOST_EXEC}/compiler/tflchef/tools/file/tflchef-file)
  set(TFLCHEF_REVERSE_PATH $ENV{BUILD_HOST_EXEC}/compiler/tflchef/tools/reverse/tflchef-reverse)
  message(STATUS "TFLCHEF_FILE_PATH = ${TFLCHEF_FILE_PATH}")
endif(DEFINED ENV{BUILD_HOST_EXEC})

nncc_find_resource(TensorFlowLiteRecipes)
set(TENSORFLOWLITERECIPES_DIR "${TensorFlowLiteRecipes_DIR}")

file(GLOB RECIPES RELATIVE ${TENSORFLOWLITERECIPES_DIR} "${TENSORFLOWLITERECIPES_DIR}/*/test.recipe")

foreach(RECIPE IN ITEMS ${RECIPES})
  get_filename_component(RECIPE_PREFIX ${RECIPE} DIRECTORY)

  set(RECIPE_SOURCE_FILE "${RECIPE_PREFIX}.recipe")
  set(RECIPE_OUTPUT_FILE "${RECIPE_PREFIX}.tflite")

  # Copy .recipe
  add_custom_command(OUTPUT ${RECIPE_SOURCE_FILE}
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                             "${TENSORFLOWLITERECIPES_DIR}/${RECIPE}" ${RECIPE_SOURCE_FILE}
                     DEPENDS "${TENSORFLOWLITERECIPES_DIR}/${RECIPE}"
                     COMMENT "Generating ${RECIPE_SOURCE_FILE}")

  # Generate .tflite
  add_custom_command(OUTPUT ${RECIPE_OUTPUT_FILE}
                     COMMAND ${TFLCHEF_FILE_PATH} ${RECIPE_SOURCE_FILE} ${RECIPE_OUTPUT_FILE}
                     DEPENDS ${TFLCHEF_FILE_PATH} ${RECIPE_SOURCE_FILE}
                     COMMENT "Generating ${RECIPE_OUTPUT_FILE}")

  list(APPEND TESTS ${RECIPE_PREFIX})
  list(APPEND TESTFILES ${RECIPE_OUTPUT_FILE})
endforeach(RECIPE)

# Add local files
file(GLOB RECIPES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*/test.recipe")

foreach(RECIPE IN ITEMS ${RECIPES})
  get_filename_component(RECIPE_PREFIX ${RECIPE} DIRECTORY)

  set(RECIPE_SOURCE_FILE "${RECIPE_PREFIX}.recipe")
  set(RECIPE_OUTPUT_FILE "${RECIPE_PREFIX}.tflite")

  # Copy .recipe
  add_custom_command(OUTPUT ${RECIPE_SOURCE_FILE}
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                             "${CMAKE_CURRENT_SOURCE_DIR}/${RECIPE}" ${RECIPE_SOURCE_FILE}
                     DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${RECIPE}"
                     COMMENT "Generating ${RECIPE_SOURCE_FILE}")

  # Generate .tflite
  add_custom_command(OUTPUT ${RECIPE_OUTPUT_FILE}
                     COMMAND ${TFLCHEF_FILE_PATH} ${RECIPE_SOURCE_FILE} ${RECIPE_OUTPUT_FILE}
                     DEPENDS ${TFLCHEF_FILE_PATH} ${RECIPE_SOURCE_FILE}
                     COMMENT "Generating ${RECIPE_OUTPUT_FILE}")

  list(APPEND TESTS ${RECIPE_PREFIX})
  list(APPEND TESTFILES ${RECIPE_OUTPUT_FILE})
endforeach(RECIPE)

# Test tflchef-reverse
file(GLOB GEN_TFLITEFILES RELATIVE ${TENSORFLOWLITERECIPES_DIR} "${TENSORFLOWLITERECIPES_DIR}/*/test.reverse")
# Note: While in development, tflchef-reverse may not handle the operator.
#       To separate this linkage scan empty test.reverse for test targets for tflchef-reverse.

foreach(TFLITEFILE IN ITEMS ${GEN_TFLITEFILES})
  get_filename_component(TFLITE_PREFIX ${TFLITEFILE} DIRECTORY)

  # file from above tflchef-file block
  # use tflite file as input of tflchef-reverse generated from tflchef-file
  set(RECIPE_OUTPUT_FILE "${TFLITE_PREFIX}.tflite")
  set(RECIPE_GEN_OUTPUT_FILE "${TFLITE_PREFIX}.gen.recipe")
  set(RECIPE_GEN_OUTPUT_FILE2 "${TFLITE_PREFIX}.gen.tflite")

  # Generate .gen.recipe from generated .tflite
  add_custom_command(OUTPUT ${RECIPE_GEN_OUTPUT_FILE}
                     COMMAND ${TFLCHEF_REVERSE_PATH} ${RECIPE_OUTPUT_FILE} ${RECIPE_GEN_OUTPUT_FILE}
                     DEPENDS ${TFLCHEF_REVERSE_PATH} ${RECIPE_OUTPUT_FILE}
                     COMMENT "Generating ${RECIPE_GEN_OUTPUT_FILE}")

  # now we are going to generate .gen.tflite from .gen.recipe
  # to check generated .gen.recipe file is correct by using it.
  # as weight values may be different, binary comparision is not acceptable.
  add_custom_command(OUTPUT ${RECIPE_GEN_OUTPUT_FILE2}
                     COMMAND ${TFLCHEF_FILE_PATH} ${RECIPE_GEN_OUTPUT_FILE} ${RECIPE_GEN_OUTPUT_FILE2}
                     DEPENDS ${TFLCHEF_FILE_PATH} ${RECIPE_GEN_OUTPUT_FILE}
                     COMMENT "Generating ${RECIPE_GEN_OUTPUT_FILE2}")

  list(APPEND TESTS ${TFLITE_PREFIX}.gen)
  list(APPEND TESTFILES ${RECIPE_GEN_OUTPUT_FILE2})
endforeach(TFLITEFILE)

# Test local tflchef-reverse
file(GLOB GEN_TFLITEFILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*/test.reverse")

foreach(TFLITEFILE IN ITEMS ${GEN_TFLITEFILES})
  get_filename_component(TFLITE_PREFIX ${TFLITEFILE} DIRECTORY)

  set(RECIPE_OUTPUT_FILE "${TFLITE_PREFIX}.tflite")
  set(RECIPE_GEN_OUTPUT_FILE "${TFLITE_PREFIX}.gen.recipe")
  set(RECIPE_GEN_OUTPUT_FILE2 "${TFLITE_PREFIX}.gen.tflite")

  # Generate .gen.recipe from generated .tflite
  add_custom_command(OUTPUT ${RECIPE_GEN_OUTPUT_FILE}
                     COMMAND ${TFLCHEF_REVERSE_PATH} ${RECIPE_OUTPUT_FILE} ${RECIPE_GEN_OUTPUT_FILE}
                     DEPENDS ${TFLCHEF_REVERSE_PATH} ${RECIPE_OUTPUT_FILE}
                     COMMENT "Generating ${RECIPE_GEN_OUTPUT_FILE}")

  add_custom_command(OUTPUT ${RECIPE_GEN_OUTPUT_FILE2}
                     COMMAND ${TFLCHEF_FILE_PATH} ${RECIPE_GEN_OUTPUT_FILE} ${RECIPE_GEN_OUTPUT_FILE2}
                     DEPENDS ${TFLCHEF_FILE_PATH} ${RECIPE_GEN_OUTPUT_FILE}
                     COMMENT "Generating ${RECIPE_GEN_OUTPUT_FILE2}")

  list(APPEND TESTS ${TFLITE_PREFIX}.gen)
  list(APPEND TESTFILES ${RECIPE_GEN_OUTPUT_FILE2})
endforeach(TFLITEFILE)

# Add a dummy target to create a target-level dependency.
# TODO Find a way to create a dependency between tflchef_test and generated testfiles.
add_custom_target(tflchef_testfiles ALL DEPENDS ${TESTFILES})

# Using mio_tflite_validate for temporary as it only calls flatbuffer validate
# TODO do testing with running the model with runtime/interpreter
# NOTE for ARM32 cross build, $<TARGET_FILE:mio_tflite2121_validate> is used as-is
#      as test should run in ARM32 device
add_test(NAME tflchef_test
         COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/runvalidate.sh"
                 $<TARGET_FILE:mio_tflite2121_validate>
                 ${TESTS})
