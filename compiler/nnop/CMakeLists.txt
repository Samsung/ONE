add_library(nnop INTERFACE)
target_include_directories(nnop INTERFACE include)
target_link_libraries(nnop INTERFACE angkor)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest QUIET)

file(GLOB_RECURSE TESTS "src/*.test.cpp")

add_executable(nnop_test ${TESTS})
target_link_libraries(nnop_test nnop)
target_link_libraries(nnop_test gtest_main)
add_test(nnop_test nnop_test)
