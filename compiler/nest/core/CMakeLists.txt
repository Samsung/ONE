file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(nest_core STATIC ${SOURCES})
set_target_properties(nest_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nest_core PUBLIC include)

file(GLOB EXAMPLE_FILES "examples/*.cpp")

foreach(EXAMPLE_FILE IN ITEMS ${EXAMPLE_FILES})
  get_filename_component(EXAMPLE_NAME ${EXAMPLE_FILE} NAME_WE)
  set(TARGET_NAME nest_IR_example_${EXAMPLE_NAME})
  add_executable(${TARGET_NAME} ${EXAMPLE_FILE})
  target_link_libraries(${TARGET_NAME} nest_core)
endforeach(EXAMPLE_FILE)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest QUIET)

add_executable(nest_core_test ${TESTS})
target_link_libraries(nest_core_test gtest_main)
target_link_libraries(nest_core_test nest_core)
add_test(nest_core_test nest_core_test)
