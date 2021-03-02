file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(tfinfo STATIC ${SOURCES})
set_target_properties(tfinfo PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tfinfo PUBLIC include)
target_link_libraries(tfinfo angkor oops)

# TODO Remove "nnkit_support_tftestinfo" later
add_library(nnkit_support_tftestinfo ALIAS tfinfo)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(tfinfo_test ${TESTS})
target_link_libraries(tfinfo_test tfinfo)
