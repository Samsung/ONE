if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

set(VIRTUALENV "${NNCC_OVERLAY_DIR}/venv_2_12_1")
set(TEST_LIST_FILE "test.lst")

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

macro(eval RECIPE PASS_OPTION)
  set(CIRCLE_FILE "${RECIPE}.circle")
  set(CIRCLE_PATH "${ARTIFACTS_BIN_PATH}/${CIRCLE_FILE}")

  set(PASS_CIRCLE_FILE "${RECIPE}.pass.circle")
  set(PASS_CIRCLE_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${PASS_CIRCLE_FILE}")

  set(DASH_PASS_OPTION "--${PASS_OPTION}")
  foreach(MORE_OPTIONS ${ARGN})
    list(APPEND DASH_PASS_OPTION "--${MORE_OPTIONS}")
  endforeach()
  # NOTE if there are two options, 'DASH_PASS_OPTION' will be like '--option_a;--option_b'
  #      add_custom_command() will translate ';' to two arguments as '--optiona_a --optionb'
  #      do not use set(DASH_PASS_OPTION "${DASH_PASS_OPTION} --${ARG}"))
  #      as this will become like '"--optiona_a --optionb"' which is one string argument

  # Generate optimized .circle
  add_custom_command(OUTPUT ${PASS_CIRCLE_OUTPUT_PATH}
    COMMAND $<TARGET_FILE:circle2circle> ${DASH_PASS_OPTION} ${CIRCLE_PATH} ${PASS_CIRCLE_OUTPUT_PATH}
    DEPENDS $<TARGET_FILE:circle2circle> ${CIRCLE_PATH}
    COMMENT "Generate ${PASS_CIRCLE_FILE} with ${DASH_PASS_OPTION}"
  )

  # depends
  list(APPEND TEST_DEPS ${PASS_CIRCLE_OUTPUT_PATH})

endmacro(eval)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

add_custom_target(luci_pass_value_py_test_files ALL DEPENDS ${TEST_DEPS})
add_dependencies(luci_pass_value_py_test_files common_artifacts_deps)

add_test(NAME luci_pass_value_py_test
  COMMAND ${VIRTUALENV}/bin/python -m pytest -sv test_luci_eval.py
        --test_list ${TEST_LIST_FILE}
        --tflite_dir ${ARTIFACTS_BIN_PATH}
        --circle_dir ${CMAKE_CURRENT_BINARY_DIR}
        --luci_eval_driver $<TARGET_FILE:luci_eval_driver>
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
