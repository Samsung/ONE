add_library(fipe INTERFACE)
target_include_directories(fipe INTERFACE include)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(fipe_test fipe.test.cpp)
target_link_libraries(fipe_test fipe)
