file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(pp STATIC ${SOURCES})
set_target_properties(pp PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pp PUBLIC include)
target_link_libraries(pp PRIVATE nncc_common)
target_link_libraries(pp PUBLIC nncc_coverage)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Google Test is mandatory for internal testing
nnas_find_package(GTest REQUIRED)

add_executable(pp_test ${TESTS})
target_link_libraries(pp_test pp)
target_link_libraries(pp_test gtest_main)
add_test(pp_test pp_test)
