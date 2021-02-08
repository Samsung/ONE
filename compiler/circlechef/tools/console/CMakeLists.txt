add_executable(circlechef Driver.cpp)
target_link_libraries(circlechef circlechef_core)
target_link_libraries(circlechef safemain)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(circlechef_test Driver.test.cpp Driver.cpp)
target_link_libraries(circlechef_test circlechef_core)
