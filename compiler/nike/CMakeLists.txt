file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(nike STATIC ${SOURCES})
target_include_directories(nike PUBLIC include)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest QUIET)

GTest_AddTest(nike_test ${TESTS})
target_link_libraries(nike_test nike)
