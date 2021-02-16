add_library(pepper_str INTERFACE)
target_include_directories(pepper_str INTERFACE include)
target_link_libraries(pepper_str INTERFACE nncc_coverage)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Google Test is mandatory for test
nnas_find_package(GTest REQUIRED)

GTest_AddTest(pepper_str_test src/pepper-str.test.cpp)
target_link_libraries(pepper_str_test pepper_str)
