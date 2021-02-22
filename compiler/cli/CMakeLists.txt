list(APPEND SOURCES "src/App.cpp")
list(APPEND TESTS "src/App.test.cpp")

add_library(cli ${SOURCES})
target_include_directories(cli PUBLIC include)

nnas_find_package(GTest QUIET)

if(NOT GTest_FOUND)
  return()
endif(NOT GTest_FOUND)

GTest_AddTEst(cli_test ${TESTS})
target_link_libraries(cli_test cli)
