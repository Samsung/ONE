file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(cwrap STATIC ${SOURCES})
set_target_properties(cwrap PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(cwrap PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(cwrap PUBLIC include)
# Let's apply nncc common compile options
# NOTE This will enable strict compilation (warnings as error).
#      Please refer to top-level CMakeLists.txt for details
target_link_libraries(cwrap PRIVATE nncc_common)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Google Test is mandatory for testing
nnas_find_package(GTest REQUIRED)

GTest_AddTest(cwrap_test ${TESTS})
target_link_libraries(cwrap_test cwrap)
