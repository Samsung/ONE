file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(mir2loco STATIC ${SOURCES})
target_include_directories(mir2loco PRIVATE src)
target_include_directories(mir2loco PUBLIC include)
target_link_libraries(mir2loco PUBLIC mir)
target_link_libraries(mir2loco PUBLIC loco)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest QUIET)

GTest_AddTest(mir2loco_test ${TESTS})
target_link_libraries(mir2loco_test mir2loco)
