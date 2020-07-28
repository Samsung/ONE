if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

unset(DALGONA_SINGLE_OP_TEST)

nnas_find_package(FlatBuffers EXACT 2.0 QUIET)
if(NOT FlatBuffers_FOUND)
  message(STATUS "Build dalgona: FAILED (missing FlatBuffers)")
  return()
endif(NOT FlatBuffers_FOUND)

macro(singleOpTest NAME)
  list(APPEND DALGONA_SINGLE_OP_TEST ${NAME})
endmacro(singleOpTest)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

unset(TEST_DEPS)

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)
get_target_property(SCHEMA_BIN_PATH mio_circle04 BINARY_DIR)

# Place test scripts in one place
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/gen_h5_random_inputs.py" "${CMAKE_CURRENT_BINARY_DIR}/gen_h5_random_inputs.py" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/SingleOperatorTest.py" "${CMAKE_CURRENT_BINARY_DIR}/SingleOperatorTest.py" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/TestUtil.py" "${CMAKE_CURRENT_BINARY_DIR}/TestUtil.py" COPYONLY)

###
### Generate test.config
###
set(TEST_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/test.config")

add_custom_command(
  OUTPUT ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E remove -f ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'DALGONA_PATH=\"$<TARGET_FILE:dalgona>\"' >> ${TEST_CONFIG}
  DEPENDS dalgona
  COMMENT "Generate test configuration"
)

###
### Generate python interface for circle schema
###
set(CIRCLE_SCHEMA_PYTHON_DIR "${CMAKE_CURRENT_BINARY_DIR}/circle")

add_custom_command(
  OUTPUT ${CIRCLE_SCHEMA_PYTHON_DIR}
  COMMAND ${CMAKE_COMMAND} -E remove_directory "${CIRCLE_SCHEMA_PYTHON_DIR}"
  COMMAND "$<TARGET_FILE:flatbuffers::flatc>" --python
  -o "${CMAKE_CURRENT_BINARY_DIR}" "${SCHEMA_BIN_PATH}/schema.fbs"
  DEPENDS flatbuffers::flatc
  COMMENT "Generate python interface for circle schema"
)

list(APPEND TEST_DEPS "${TEST_CONFIG}" "${CIRCLE_SCHEMA_PYTHON_DIR}")

# This enforces CMake to generate all the dependencies during "build" phase
add_custom_target(dalgona_test_deps ALL DEPENDS ${TEST_DEPS})

# Run tests
add_test(
  NAME dalgona_single_op_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/test_single_op.sh"
          "${TEST_CONFIG}"
          "${ARTIFACTS_BIN_PATH}"
          "${NNCC_OVERLAY_DIR}/venv_2_8_0"
          ${DALGONA_SINGLE_OP_TEST}
)
