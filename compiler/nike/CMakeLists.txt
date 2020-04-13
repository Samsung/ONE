file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(nike STATIC ${SOURCES})
target_include_directories(nike PUBLIC include)

nnas_find_package(GTest QUIET)

if(NOT GTest_FOUND)
  return()
endif(NOT GTest_FOUND)

GTest_AddTest(nike_test ${TESTS})
target_link_libraries(nike_test nike)
