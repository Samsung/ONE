if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

file(GLOB_RECURSE TESTS "*.cc")

add_executable(npud_gtest ${TESTS})

set_target_properties(npud_gtest PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(npud_gtest PUBLIC ${NPUD_INCLUDE_DIRS})
target_include_directories(npud_gtest PUBLIC ${GLIB2.0_INCLUDE_DIRS})
target_link_libraries(npud_gtest PRIVATE ${GLIB2.0_LIBRARIES})
target_link_libraries(npud_gtest PRIVATE Threads::Threads)
target_link_libraries(npud_gtest PRIVATE npud_core)
target_link_libraries(npud_gtest PRIVATE gtest_main)

install(TARGETS npud_gtest DESTINATION npud-gtest)
