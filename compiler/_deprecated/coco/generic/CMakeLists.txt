file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(coco_generic SHARED ${SOURCES})
target_include_directories(coco_generic PUBLIC include)
target_link_libraries(coco_generic PUBLIC coco_core)
target_link_libraries(coco_generic PRIVATE nncc_common)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Google Test is required for internal testing
nnas_find_package(GTest REQUIRED)

GTest_AddTest(coco_generic_test ${TESTS})
target_link_libraries(coco_generic_test coco_generic)
