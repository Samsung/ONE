if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

get_target_property(ARTIFACTS_PATH testDataGenerator BINARY_DIR)
get_target_property(CIRCLE_INTERPRETER_PATH circle-interpreter BINARY_DIR)
set(CIRCLE_INTERPRETER_PATH "${CIRCLE_INTERPRETER_PATH}/circle-interpreter")

nnas_find_package(GTest REQUIRED)

file(GLOB_RECURSE TESTS "src/*.test.cpp")
GTest_AddTest(circle-interpreter-test ${TESTS})

set_tests_properties(circle-interpreter-test
                     PROPERTIES
                     ENVIRONMENT "ARTIFACTS_PATH=${ARTIFACTS_PATH};CIRCLE_INTERPRETER_PATH=${CIRCLE_INTERPRETER_PATH}"
                     )
