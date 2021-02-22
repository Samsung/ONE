file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(locop STATIC ${SOURCES})
set_target_properties(locop PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(locop PUBLIC include)
target_link_libraries(locop PUBLIC loco)
# Let's apply nncc common compile options
#
# NOTE This will enable strict compilation (warnings as error).
#      Please refer to the top-level CMakeLists.txt for details
target_link_libraries(locop PRIVATE nncc_common)
target_link_libraries(locop PUBLIC nncc_coverage)
target_link_libraries(locop PRIVATE pp)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Google Test is mandatory for internal testing
nnas_find_package(GTest REQUIRED)

GTest_AddTest(locop_test ${TESTS})
target_link_libraries(locop_test locop)
