file(GLOB_RECURSE TESTS "src/*.test.cpp")

add_library(stdex INTERFACE)
target_include_directories(stdex INTERFACE include)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Google Test is mandatory for test
nnas_find_package(GTest REQUIRED)

add_executable(stdex_test ${TESTS})
target_link_libraries(stdex_test stdex)
target_link_libraries(stdex_test gtest_main)
add_test(stdex_test stdex_test)
