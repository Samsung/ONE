if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

get_target_property(ARTIFACTS_PATH testDataGenerator BINARY_DIR)
get_target_property(CIRCLE_INTERPRETER_PATH circle-interpreter BINARY_DIR)
set(CIRCLE_INTERPRETER_PATH "${CIRCLE_INTERPRETER_PATH}/circle-interpreter")

nnas_find_package(GTest REQUIRED)

file(GLOB_RECURSE TESTS "src/*.test.cpp")
GTest_AddTest(circle-interpreter-test ${TESTS})

# circle-interpreter-test uses input data generated during luci_value_test
if(NOT CMAKE_CROSSCOMPILING)
  set_tests_properties(circle-interpreter-test
                      PROPERTIES
                      DEPENDS luci_value_test
                      ENVIRONMENT "ARTIFACTS_PATH=${ARTIFACTS_PATH};CIRCLE_INTERPRETER_PATH=${CIRCLE_INTERPRETER_PATH}"
                      )
else(NOT CMAKE_CROSSCOMPILING)
  set_tests_properties(circle-interpreter-test
                      PROPERTIES
                      DEPENDS luci_value_cross_test
                      ENVIRONMENT "ARTIFACTS_PATH=${ARTIFACTS_PATH};CIRCLE_INTERPRETER_PATH=${CIRCLE_INTERPRETER_PATH}"
                      )
endif(NOT CMAKE_CROSSCOMPILING)
