if(NOT ENABLE_TEST)
    return()
endif(NOT ENABLE_TEST)

# Disable test if minmax-embedder does not exist
if (NOT TARGET minmax_embedder)
    message(STATUS "minmax-embedder-test is disabled as minmax-embedder was not built.")
    return()
endif(NOT TARGET minmax_embedder)

add_subdirectory(gen)

# Add tests
unset(MINMAX_EMBEDDER_VALUE_TESTS)

macro(addTest NAME)
  list(APPEND MINMAX_EMBEDDER_VALUE_TESTS ${NAME})
endmacro(addTest)

include("test.lst")
include("test.local.lst" OPTIONAL)

unset(TEST_DEPS)

# Generate test.config
set(TEST_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/test.config")

add_custom_command(
  OUTPUT ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E remove -f ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'ARTIFACTS_PATH=\"$<TARGET_FILE_DIR:testDataGenerator>\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'MINMAX_DATA_GEN=\"$<TARGET_FILE:minmax_data_gen>\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'MINMAX_EMBEDDER=\"$<TARGET_FILE:minmax_embedder_driver>\"' >> ${TEST_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E echo 'CIRCLEDUMP=\"$<TARGET_FILE:circledump>\"' >> ${TEST_CONFIG}
  DEPENDS testDataGenerator
  DEPENDS minmax_data_gen
  DEPENDS minmax_embedder_driver
  DEPENDS circledump
  COMMENT "Generate test configuration"
)

list(APPEND TEST_DEPS "${TEST_CONFIG}")

# This enforces CMake to generate all the dependencies during "build" phase
add_custom_target(minmax_embedder_value_test_deps ALL DEPENDS ${TEST_DEPS})

# Run tests
add_test(
  NAME minmax_embedder_value_test
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/testall.sh"
        "${TEST_CONFIG}"
        ${MINMAX_EMBEDDER_VALUE_TESTS}
)
