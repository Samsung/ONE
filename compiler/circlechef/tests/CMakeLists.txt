nncc_find_resource(CircleRecipes)
set(CIRCLERECIPES_DIR "${CircleRecipes_DIR}")

file(GLOB RECIPES RELATIVE ${CIRCLERECIPES_DIR} "${CIRCLERECIPES_DIR}/*/test.recipe")

foreach(RECIPE IN ITEMS ${RECIPES})
  get_filename_component(RECIPE_PREFIX ${RECIPE} DIRECTORY)

  set(RECIPE_SOURCE_FILE "${RECIPE_PREFIX}.recipe")
  set(RECIPE_OUTPUT_FILE "${RECIPE_PREFIX}.circle")

  # Copy .recipe
  add_custom_command(OUTPUT ${RECIPE_SOURCE_FILE}
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                             "${CIRCLERECIPES_DIR}/${RECIPE}" ${RECIPE_SOURCE_FILE}
                     DEPENDS "${CIRCLERECIPES_DIR}/${RECIPE}"
                     COMMENT "Generating ${RECIPE_SOURCE_FILE}")

  # Generate .circle
  add_custom_command(OUTPUT ${RECIPE_OUTPUT_FILE}
                     COMMAND circlechef-file ${RECIPE_SOURCE_FILE} ${RECIPE_OUTPUT_FILE}
                     DEPENDS circlechef-file ${RECIPE_SOURCE_FILE}
                     COMMENT "Generating ${RECIPE_OUTPUT_FILE}")

  list(APPEND TESTS ${RECIPE_PREFIX})
  list(APPEND TESTFILES ${RECIPE_OUTPUT_FILE})
endforeach(RECIPE)

# Add local files
file(GLOB RECIPES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*/test.recipe")

foreach(RECIPE IN ITEMS ${RECIPES})
  get_filename_component(RECIPE_PREFIX ${RECIPE} DIRECTORY)

  set(RECIPE_SOURCE_FILE "${RECIPE_PREFIX}.recipe")
  set(RECIPE_OUTPUT_FILE "${RECIPE_PREFIX}.circle")

  # Copy .recipe
  add_custom_command(OUTPUT ${RECIPE_SOURCE_FILE}
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                             "${CMAKE_CURRENT_SOURCE_DIR}/${RECIPE}" ${RECIPE_SOURCE_FILE}
                     DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${RECIPE}"
                     COMMENT "Generating ${RECIPE_SOURCE_FILE}")

  # Generate .circle
  add_custom_command(OUTPUT ${RECIPE_OUTPUT_FILE}
                     COMMAND circlechef-file ${RECIPE_SOURCE_FILE} ${RECIPE_OUTPUT_FILE}
                     DEPENDS circlechef-file ${RECIPE_SOURCE_FILE}
                     COMMENT "Generating ${RECIPE_OUTPUT_FILE}")

  list(APPEND TESTS ${RECIPE_PREFIX})
  list(APPEND TESTFILES ${RECIPE_OUTPUT_FILE})
endforeach(RECIPE)

#Test circlechef-reverse
file(GLOB GEN_CIRCLEFILES RELATIVE ${CIRCLERECIPES_DIR} "${CIRCLERECIPES_DIR}/*/test.reverse")
# Note: While in development, circlechef-reverse may not handle the operator.
#       To separate this linkage scan empty test.reverse for test targets for circlechef-reverse.

foreach(CIRCLEFILE IN ITEMS ${GEN_CIRCLEFILES})
  get_filename_component(CIRCLE_PREFIX ${CIRCLEFILE} DIRECTORY)

  # file from above circlechef-file block
  # use circle file as input of circlechef-reverse generated from circlechef-file
  set(RECIPE_OUTPUT_FILE "${CIRCLE_PREFIX}.circle")
  set(RECIPE_GEN_OUTPUT_FILE "${CIRCLE_PREFIX}.gen.recipe")
  set(RECIPE_GEN_OUTPUT_FILE2 "${CIRCLE_PREFIX}.gen.circle")

  # Generate .gen.recipe from generated .circle
  add_custom_command(OUTPUT ${RECIPE_GEN_OUTPUT_FILE}
                     COMMAND circlechef-reverse ${RECIPE_OUTPUT_FILE} ${RECIPE_GEN_OUTPUT_FILE}
                     DEPENDS circlechef-reverse ${RECIPE_OUTPUT_FILE}
                     COMMENT "Generating ${RECIPE_GEN_OUTPUT_FILE}")

  # now we are going to generate .gen.circle from .gen.recipe
  # to check generated .gen.recipe file is correct by using it.
  # as weight values may be different, binary comparision is not acceptable.
  add_custom_command(OUTPUT ${RECIPE_GEN_OUTPUT_FILE2}
                     COMMAND circlechef-file ${RECIPE_GEN_OUTPUT_FILE} ${RECIPE_GEN_OUTPUT_FILE2}
                     DEPENDS circlechef-file ${RECIPE_GEN_OUTPUT_FILE}
                     COMMENT "Generating ${RECIPE_GEN_OUTPUT_FILE2}")

  list(APPEND TESTS ${CIRCLE_PREFIX}.gen)
  list(APPEND TESTFILES ${RECIPE_GEN_OUTPUT_FILE2})
endforeach(CIRCLEFILE)

# Test local circlechef-reverse
file(GLOB GEN_CIRCLEFILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*/test.reverse")

foreach(CIRCLEFILE IN ITEMS ${GEN_CIRCLEFILES})
  get_filename_component(CIRCLE_PREFIX ${CIRCLEFILE} DIRECTORY)

  set(RECIPE_OUTPUT_FILE "${CIRCLE_PREFIX}.circle")
  set(RECIPE_GEN_OUTPUT_FILE "${CIRCLE_PREFIX}.gen.recipe")
  set(RECIPE_GEN_OUTPUT_FILE2 "${CIRCLE_PREFIX}.gen.circle")

  # Generate .gen.recipe from generated .circle
  add_custom_command(OUTPUT ${RECIPE_GEN_OUTPUT_FILE}
                     COMMAND circlechef-reverse ${RECIPE_OUTPUT_FILE} ${RECIPE_GEN_OUTPUT_FILE}
                     DEPENDS circlechef-reverse ${RECIPE_OUTPUT_FILE}
                     COMMENT "Generating ${RECIPE_GEN_OUTPUT_FILE}")

  add_custom_command(OUTPUT ${RECIPE_GEN_OUTPUT_FILE2}
                     COMMAND circlechef-file ${RECIPE_GEN_OUTPUT_FILE} ${RECIPE_GEN_OUTPUT_FILE2}
                     DEPENDS circlechef-file ${RECIPE_GEN_OUTPUT_FILE}
                     COMMENT "Generating ${RECIPE_GEN_OUTPUT_FILE2}")

  list(APPEND TESTS ${CIRCLE_PREFIX}.gen)
  list(APPEND TESTFILES ${RECIPE_GEN_OUTPUT_FILE2})
endforeach(CIRCLEFILE)

# Add a dummy target to create a target-level dependency.
# TODO Find a way to create a dependency between circlechef_test and generated testfiles.
add_custom_target(circlechef_testfiles ALL DEPENDS ${TESTFILES})

# Using circle_verify for temporary as it only calls flatbuffer validate
# TODO do testing with running the model with runtime/interpreter
add_test(NAME circlechef_test
         COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/runvalidate.sh"
                 $<TARGET_FILE:circle-verify>
                 ${TESTS})
