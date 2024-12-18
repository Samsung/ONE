if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

get_target_property(CIRCLE_INPUT_NAMES_PATH circle_input_names BINARY_DIR)
set(CIRCLE_INPUT_NAMES_PATH "${CIRCLE_INPUT_NAMES_PATH}/circle_input_names")

nnas_find_package(GTest REQUIRED)
nnas_find_package(Jsoncpp REQUIRED)

file(GLOB TESTS_SOURCE "src/*.test.cpp")

GTest_AddTest(circle-input-names-test ${TESTS_SOURCE})

set_tests_properties(circle-input-names-test
                    PROPERTIES
                    ENVIRONMENT "CIRCLE_INPUT_NAMES_PATH=${CIRCLE_INPUT_NAMES_PATH}")
