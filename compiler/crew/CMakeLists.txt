file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(crew STATIC ${SOURCES})
target_include_directories(crew PRIVATE src)
target_include_directories(crew PUBLIC include)
target_link_libraries(crew PRIVATE foder)
target_link_libraries(crew PRIVATE nncc_common)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(crew_test ${TESTS})
target_include_directories(crew_test PRIVATE src)
target_link_libraries(crew_test nncc_common)
target_link_libraries(crew_test crew)
