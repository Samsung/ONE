add_library(oops INTERFACE)
target_include_directories(oops INTERFACE include)
target_link_libraries(oops INTERFACE pepper_str)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(oops_test src/oops.test.cpp)
target_link_libraries(oops_test oops)
