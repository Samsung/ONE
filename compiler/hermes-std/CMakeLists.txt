file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(hermes_std STATIC ${SOURCES})
set_target_properties(hermes_std PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(hermes_std PUBLIC include)
target_link_libraries(hermes_std PUBLIC hermes)
target_link_libraries(hermes_std PRIVATE pepper_strcast)
# Let's apply nncc common compile options
#
# NOTE This will enable strict compilation (warnings as error).
#      Please refer to the top-level CMakeLists.txt for details
target_link_libraries(hermes_std PRIVATE nncc_common)
target_link_libraries(hermes_std PUBLIC nncc_coverage)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Google Test is mandatory for internal testing
nnas_find_package(GTest REQUIRED)

GTest_AddTest(hermes_std_test ${TESTS})
target_link_libraries(hermes_std_test hermes_std)
