file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(morph STATIC ${SOURCES})
set_target_properties(morph PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(morph PUBLIC include)
target_link_libraries(morph PRIVATE nncc_common)
target_link_libraries(morph PUBLIC angkor)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest QUIET)

add_executable(morph_test ${TESTS})
target_link_libraries(morph_test morph)
target_link_libraries(morph_test gtest_main)
add_test(morph_test morph_test)
