if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

unset(DALGONA_SINGLE_OP_TEST)

macro(singleOpTest NAME)
  list(APPEND DALGONA_SINGLE_OP_TEST ${NAME})
endmacro(singleOpTest)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)

unset(TEST_DEPS)

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

# Place test scripts in one place
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/GenH5RandomInputs.py" "${CMAKE_CURRENT_BINARY_DIR}/GenH5RandomInputs.py" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/SingleOperatorTest.py" "${CMAKE_CURRENT_BINARY_DIR}/SingleOperatorTest.py" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/RandomDataGenerator.py" "${CMAKE_CURRENT_BINARY_DIR}/RandomDataGenerator.py" COPYONLY)
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

# Import pics module
get_target_property(PICS_BIN_PATH pics BINARY_DIR)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/circle
                   COMMAND ${CMAKE_COMMAND} -E create_symlink
                   ${PICS_BIN_PATH}/circle ${CMAKE_CURRENT_BINARY_DIR}/circle)

list(APPEND TEST_DEPS "${TEST_CONFIG}" "${CMAKE_CURRENT_BINARY_DIR}/circle")

# This enforces CMake to generate all the dependencies during "build" phase
add_custom_target(dalgona_test_deps ALL DEPENDS ${TEST_DEPS})

# Run tests
add_test(
  NAME dalgona_single_op_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/TestSingleOp.sh"
          "${TEST_CONFIG}"
          "${ARTIFACTS_BIN_PATH}"
          "${NNCC_OVERLAY_DIR}/venv_2_12_1"
          ${DALGONA_SINGLE_OP_TEST}
)
