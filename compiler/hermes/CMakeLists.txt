file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(hermes STATIC ${SOURCES})
set_target_properties(hermes PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(hermes PUBLIC include)
# Let's apply nncc common compile options
#
# NOTE This will enable strict compilation (warnings as error).
#      Please refer to the top-level CMakeLists.txt for details
target_link_libraries(hermes PRIVATE nncc_common)
target_link_libraries(hermes PUBLIC nncc_coverage)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Google Test is mandatory for internal testing
nnas_find_package(GTest REQUIRED)

add_executable(hermes_test ${TESTS})
target_link_libraries(hermes_test gtest_main)
target_link_libraries(hermes_test hermes)

add_test(hermes_test hermes_test)
