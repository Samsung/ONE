if(NOT TARGET nnkit-run)
  return()
endif(NOT TARGET nnkit-run)

if(NOT TARGET nnkit_tflite_backend)
  return()
endif(NOT TARGET nnkit_tflite_backend)

nncc_find_resource(TensorFlowLiteRecipes)
set(TENSORFLOWLITERECIPES_DIR "${TensorFlowLiteRecipes_DIR}")

file(GLOB RECIPES RELATIVE ${TENSORFLOWLITERECIPES_DIR} "${TENSORFLOWLITERECIPES_DIR}/*/test.recipe")

foreach(RECIPE IN ITEMS ${RECIPES})
  get_filename_component(RECIPE_PREFIX ${RECIPE} DIRECTORY)

  set(RECIPE_SOURCE_FILE "${RECIPE_PREFIX}.recipe")
  set(RECIPE_SOURCE_TARGET tflchef_${RECIPE_PREFIX}_recipe)

  set(RECIPE_OUTPUT_FILE "${RECIPE_PREFIX}.tflite")
  set(RECIPE_OUTPUT_TARGET tflchef_${RECIPE_PREFIX}_tflite)

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

  list(APPEND TESTS ${RECIPE_PREFIX})
endforeach(RECIPE)

# Add local files
file(GLOB RECIPES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*/test.recipe")

foreach(RECIPE IN ITEMS ${RECIPES})
  get_filename_component(RECIPE_PREFIX ${RECIPE} DIRECTORY)

  set(RECIPE_SOURCE_FILE "${RECIPE_PREFIX}.recipe")
  set(RECIPE_SOURCE_TARGET tflchef_${RECIPE_PREFIX}_recipe)

  set(RECIPE_OUTPUT_FILE "${RECIPE_PREFIX}.tflite")
  set(RECIPE_OUTPUT_TARGET tflchef_${RECIPE_PREFIX}_tflite)

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

  list(APPEND TESTS ${RECIPE_PREFIX})
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
  set(RECIPE_OUTPUT_TARGET tflchef_${TFLITE_PREFIX}_tflite)

  set(RECIPE_GEN_OUTPUT_FILE "${TFLITE_PREFIX}.gen.recipe")
  set(RECIPE_GEN_OUTPUT_TARGET tflchef_${TFLITE_PREFIX}_gen_recipe)

  # Generate .gen.recipe from generated .tflite
  add_custom_target(${RECIPE_GEN_OUTPUT_TARGET}
                    ALL $<TARGET_FILE:tflchef-reverse> ${RECIPE_OUTPUT_FILE} ${RECIPE_GEN_OUTPUT_FILE}
                    DEPENDS ${RECIPE_OUTPUT_TARGET}
                    COMMENT "Generate ${TFLITE_PREFIX}.gen.recipe")

  # now we are going to generate .gen.tflite from .gen.recipe
  # to check generated .gen.recipe file is correct by using it.
  # as weight values may be different, binary comparision is not acceptable.
  set(RECIPE_GEN_OUTPUT_FILE2 "${TFLITE_PREFIX}.gen.tflite")
  set(RECIPE_GEN_OUTPUT_TARGET2 tflchef_${TFLITE_PREFIX}_gen_tflite)

  add_custom_target(${RECIPE_GEN_OUTPUT_TARGET2}
                    ALL $<TARGET_FILE:tflchef-file> ${RECIPE_GEN_OUTPUT_FILE} ${RECIPE_GEN_OUTPUT_FILE2}
                    DEPENDS ${RECIPE_GEN_OUTPUT_TARGET}
                    COMMENT "Generate ${TFLITE_PREFIX}.gen.tflite")

  list(APPEND TESTS ${TFLITE_PREFIX}.gen)
endforeach(TFLITEFILE)

# Test local tflchef-reverse
file(GLOB GEN_TFLITEFILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*/test.reverse")

foreach(TFLITEFILE IN ITEMS ${GEN_TFLITEFILES})
  get_filename_component(TFLITE_PREFIX ${TFLITEFILE} DIRECTORY)

  set(RECIPE_OUTPUT_FILE "${TFLITE_PREFIX}.tflite")
  set(RECIPE_OUTPUT_TARGET tflchef_${TFLITE_PREFIX}_tflite)

  set(RECIPE_GEN_OUTPUT_FILE "${TFLITE_PREFIX}.gen.recipe")
  set(RECIPE_GEN_OUTPUT_TARGET tflchef_${TFLITE_PREFIX}_gen_recipe)

  # Generate .gen.recipe from generated .tflite
  add_custom_target(${RECIPE_GEN_OUTPUT_TARGET}
                    ALL $<TARGET_FILE:tflchef-reverse> ${RECIPE_OUTPUT_FILE} ${RECIPE_GEN_OUTPUT_FILE}
                    DEPENDS ${RECIPE_OUTPUT_TARGET}
                    COMMENT "Generate ${TFLITE_PREFIX}.gen.recipe")

  set(RECIPE_GEN_OUTPUT_FILE2 "${TFLITE_PREFIX}.gen.tflite")
  set(RECIPE_GEN_OUTPUT_TARGET2 tflchef_${TFLITE_PREFIX}_gen_tflite)

  add_custom_target(${RECIPE_GEN_OUTPUT_TARGET2}
                    ALL $<TARGET_FILE:tflchef-file> ${RECIPE_GEN_OUTPUT_FILE} ${RECIPE_GEN_OUTPUT_FILE2}
                    DEPENDS ${RECIPE_GEN_OUTPUT_TARGET}
                    COMMENT "Generate ${TFLITE_PREFIX}.gen.tflite")

  list(APPEND TESTS ${TFLITE_PREFIX}.gen)
endforeach(TFLITEFILE)

add_test(NAME tflchef_test
         COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/runall.sh"
                 $<TARGET_FILE:nnkit-run>
                 $<TARGET_FILE:nnkit_tflite_backend>
                 "${CMAKE_CURRENT_BINARY_DIR}"
                 ${TESTS})
