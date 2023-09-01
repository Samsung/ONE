file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(circlechef_core STATIC ${SOURCES})
target_include_directories(circlechef_core PUBLIC include)
target_include_directories(circlechef_core PRIVATE src)
target_link_libraries(circlechef_core PUBLIC circlechef_proto)
target_link_libraries(circlechef_core PUBLIC circlechef_log)
target_link_libraries(circlechef_core PUBLIC mio_circle06)
target_link_libraries(circlechef_core PUBLIC souschef)
target_link_libraries(circlechef_core PRIVATE nncc_coverage)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(circlechef_core_test ${TESTS})
target_include_directories(circlechef_core_test PRIVATE src)
target_link_libraries(circlechef_core_test circlechef_core)
target_link_libraries(circlechef_core_test nncc_coverage)
