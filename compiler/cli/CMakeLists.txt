list(APPEND SOURCES "src/App.cpp")
list(APPEND TESTS "src/App.test.cpp")

add_library(cli ${SOURCES})
target_include_directories(cli PUBLIC include)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest QUIET)

GTest_AddTEst(cli_test ${TESTS})
target_link_libraries(cli_test cli)
