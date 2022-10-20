if(NOT BUILD_NPUD)
  return()
endif(NOT BUILD_NPUD)

nnfw_find_package(GLib2.0 REQUIRED)

file(GLOB_RECURSE SOURCES "*.cc")
file(GLOB_RECURSE TESTS "tests/*.test.cc")
list(REMOVE_ITEM SOURCES ${TESTS})

add_executable(npud ${SOURCES})
set_target_properties(npud PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(npud PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(npud PUBLIC ${GLIB2.0_INCLUDE_DIRS})
target_link_libraries(npud PRIVATE nnfw_lib_misc)
target_link_libraries(npud PRIVATE ${GLIB2.0_LIBRARIES})
target_link_libraries(npud PRIVATE ${LIB_PTHREAD})

if(ENVVAR_NPUD_CONFIG)
  target_compile_definitions(npud PRIVATE ENVVAR_FOR_DEFAULT_CONFIG)
endif(ENVVAR_NPUD_CONFIG)

install(TARGETS npud DESTINATION bin)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

list(FILTER SOURCES EXCLUDE REGEX ".*main.cc$")

add_executable(npud_test ${TESTS} ${SOURCES})
set_target_properties(npud_test PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(npud_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(npud_test PUBLIC ${GLIB2.0_INCLUDE_DIRS})
target_link_libraries(npud_test PRIVATE nnfw_lib_misc)
target_link_libraries(npud_test PRIVATE ${GLIB2.0_LIBRARIES})
target_link_libraries(npud_test PRIVATE ${LIB_PTHREAD})
target_link_libraries(npud_test PRIVATE gtest_main dl)

install(TARGETS npud_test DESTINATION unittest)
