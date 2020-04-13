file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(kuma STATIC ${SOURCES})
set_target_properties(kuma PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(kuma PUBLIC include)
target_link_libraries(kuma PRIVATE nncc_common)
target_link_libraries(kuma PUBLIC nncc_coverage)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Google Test is mandatory for test
nnas_find_package(GTest REQUIRED)

GTest_AddTest(kuma_test ${TESTS})
target_link_libraries(kuma_test kuma)
