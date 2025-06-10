set(SOURCE "src/circle-input-names.cpp")

add_executable(circle_input_names ${SOURCE})
target_link_libraries(circle_input_names PRIVATE luci_logex)
target_link_libraries(circle_input_names PRIVATE luci_lang)
target_link_libraries(circle_input_names PRIVATE crew)
install(TARGETS circle_input_names DESTINATION bin)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

get_target_property(CIRCLE_INPUT_NAMES_PATH circle_input_names BINARY_DIR)
set(CIRCLE_INPUT_NAMES_PATH "${CIRCLE_INPUT_NAMES_PATH}/circle_input_names")

nnas_find_package(GTest REQUIRED)

set(TEST_SOURCE "src/circle-input-names.test.cpp")

GTest_AddTest(circle-input-names-test ${TEST_SOURCE})

set_tests_properties(circle-input-names-test
                    PROPERTIES
                    ENVIRONMENT "CIRCLE_INPUT_NAMES_PATH=${CIRCLE_INPUT_NAMES_PATH}")
