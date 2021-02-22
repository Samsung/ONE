file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

###
### enco_core is built as a shared library to support "interactive debugging".
###
### interactive debugging helpers are stripped during linking when enco_core is
### built as a static library
###
add_library(enco_core SHARED ${SOURCES})
target_include_directories(enco_core PRIVATE src)
target_include_directories(enco_core PUBLIC include)
target_link_libraries(enco_core PUBLIC enco_intf_cmdline)
target_link_libraries(enco_core PUBLIC coco_core)
target_link_libraries(enco_core PUBLIC coco_generic)
# These libraries are linked for internal use, and thus does not appear in public headers.
target_link_libraries(enco_core PRIVATE pp)
target_link_libraries(enco_core PRIVATE morph)
# Let's use nncc project-wide build options
target_link_libraries(enco_core PRIVATE nncc_common)

nnas_find_package(GTest QUIET)

if(NOT GTest_FOUND)
  return()
endif(NOT GTest_FOUND)

add_executable(enco_core_test ${TESTS})
target_include_directories(enco_core_test PRIVATE src)
target_link_libraries(enco_core_test gtest_main)
target_link_libraries(enco_core_test enco_core)
target_link_libraries(enco_core_test morph)
add_test(enco_core_test enco_core_test)
