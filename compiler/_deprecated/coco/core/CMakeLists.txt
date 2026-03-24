file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(coco_core SHARED ${SOURCES})
target_include_directories(coco_core PUBLIC include)
# NOTE Some coco_core PUBLIC headers include angkor headers
target_link_libraries(coco_core PUBLIC angkor)
target_link_libraries(coco_core PRIVATE pepper_assert)
# Let's apply nncc common compile options
# NOTE This will enable strict compilation (warnings as error).
#      Please refer to top-level CMakeLists.txt for details
target_link_libraries(coco_core PRIVATE nncc_common)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Google Test is required for internal testing
nnas_find_package(GTest REQUIRED)

GTest_AddTest(coco_core_test ${TESTS})
target_link_libraries(coco_core_test coco_core)
