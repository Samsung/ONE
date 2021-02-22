file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(locomotiv STATIC ${SOURCES})
set_target_properties(locomotiv PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(locomotiv PUBLIC include)
target_include_directories(locomotiv PRIVATE src)
target_link_libraries(locomotiv PUBLIC loco)
target_link_libraries(locomotiv PUBLIC angkor)
# Let's apply nncc common compile options
#
# NOTE This will enable strict compilation (warnings as error).
#      Please refer to the top-level CMakeLists.txt for details
target_link_libraries(locomotiv PRIVATE nncc_common)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Google Test is mandatory for internal testing
nnas_find_package(GTest REQUIRED)

GTest_AddTest(locomotiv_test ${TESTS})
target_include_directories(locomotiv_test PRIVATE src)
target_link_libraries(locomotiv_test locomotiv)

add_test(locomotiv_test locomotiv_test)
