option(ONECO_VALUE_PBTXT_TEST "Enable oneco value test for pbtxt input model" ON)

if(NOT ONECO_VALUE_PBTXT_TEST)
  return()
endif(NOT ONECO_VALUE_PBTXT_TEST)

if(NOT TARGET onnxkit)
  message(STATUS "oneco: Skip test material preparation as onnxkit is not defined")
  return()
endif(NOT TARGET onnxkit)

#
# Copy [Testcase]/test.pbtxt to Testcase.pbtxt in binary folder
# Encode Testcase.pbtxt to Testcase.pb
#
set(TEST_PBTXT_FILE "test.pbtxt")
set(TEST_REPO "${CMAKE_CURRENT_SOURCE_DIR}") # Where to find tests
set(TEST_SPACE "${CMAKE_CURRENT_BINARY_DIR}") # Where to run tests

file(
  GLOB PBTXTFILES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "*/${TEST_PBTXT_FILE}")

unset(TEST_DEPS)

foreach(PBTXTFILE IN ITEMS ${PBTXTFILES})
  get_filename_component(DIR_NAME ${PBTXTFILE} DIRECTORY)

  set(PBTXT_SOURCE_FILE "${DIR_NAME}.pbtxt")
  set(PBTXT_SOURCE_PATH "${TEST_SPACE}/${PBTXT_SOURCE_FILE}")

  set(PB_OUTPUT_FILE "${DIR_NAME}.pb")
  set(PB_OUTPUT_PATH "${TEST_SPACE}/${PB_OUTPUT_FILE}")

  # Copy files
  add_custom_command(
    OUTPUT ${PBTXT_SOURCE_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy "${TEST_REPO}/${PBTXTFILE}" "${PBTXT_SOURCE_PATH}"
    COMMENT "Copy ${PBTXT_SOURCE_FILE}"
    DEPENDS "${TEST_REPO}/${PBTXTFILE}")

  # Use onnxkit to encode
  add_custom_command(
    OUTPUT ${PB_OUTPUT_PATH}
    COMMAND $<TARGET_FILE:onnxkit> encode ${PBTXT_SOURCE_PATH} ${PB_OUTPUT_PATH}
    DEPENDS ${PBTXT_SOURCE_PATH}
    COMMENT "Generate ${PB_OUTPUT_FILE}")

  list(APPEND TEST_DEPS "${PB_OUTPUT_PATH}")
endforeach(PBTXTFILE)

# This target enforces CMake to generate all the dependencies during "build" phase
add_custom_target(oneco_value_pbtxt_test_deps ALL DEPENDS ${TEST_DEPS})
