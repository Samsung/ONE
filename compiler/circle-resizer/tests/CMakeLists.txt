if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

list(APPEND CIRCLE_RESIZER_TEST_SOURCES ShapeParser.test.cpp)

nnas_find_package(GTest REQUIRED)
GTest_AddTest(circle_resizer_unit_test ${CIRCLE_RESIZER_TEST_SOURCES})
target_link_libraries(circle_resizer_unit_test circle_resizer_core)
