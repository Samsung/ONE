add_library(bino INTERFACE)
target_include_directories(bino INTERFACE include)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Google Test is mandatory for internal testing
nnas_find_package(GTest REQUIRED)

file(GLOB_RECURSE TESTS "tests/*.cpp")

GTest_AddTest(bino_test ${TESTS})
target_link_libraries(bino_test bino)
