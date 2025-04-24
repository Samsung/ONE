if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

list(APPEND CIRCLE_RESIZER_TEST_SOURCES Shape.test.cpp)
list(APPEND CIRCLE_RESIZER_TEST_SOURCES ShapeParser.test.cpp)
list(APPEND CIRCLE_RESIZER_TEST_SOURCES CircleModel.test.cpp)
list(APPEND CIRCLE_RESIZER_TEST_SOURCES ModelEditor.test.cpp)

nnas_find_package(GTest REQUIRED)
GTest_AddTest(circle_resizer_unit_test ${CIRCLE_RESIZER_TEST_SOURCES})
target_link_libraries(circle_resizer_unit_test circle_resizer_core)

get_target_property(ARTIFACTS_PATH testDataGenerator BINARY_DIR)
set_tests_properties(circle_resizer_unit_test
                     PROPERTIES
                     ENVIRONMENT "ARTIFACTS_PATH=${ARTIFACTS_PATH}")
