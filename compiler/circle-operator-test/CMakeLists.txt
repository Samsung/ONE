if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

get_target_property(ARTIFACTS_PATH testDataGenerator BINARY_DIR)
get_target_property(CIRCLE_OPERATOR_PATH circle-operator BINARY_DIR)
set(CIRCLE_OPERATOR_PATH "${CIRCLE_OPERATOR_PATH}/circle-operator")

nnas_find_package(GTest REQUIRED)

file(GLOB_RECURSE TESTS "src/*.test.cpp")

GTest_AddTest(circle-operator-test ${TESTS})

set_tests_properties(
  circle-operator-test PROPERTIES ENVIRONMENT
                                  "ARTIFACTS_PATH=${ARTIFACTS_PATH};CIRCLE_OPERATOR_PATH=${CIRCLE_OPERATOR_PATH}")
