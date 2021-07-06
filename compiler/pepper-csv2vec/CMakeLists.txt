file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(pepper_csv2vec STATIC ${SOURCES})
set_target_properties(pepper_csv2vec PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pepper_csv2vec PUBLIC include)
target_link_libraries(pepper_csv2vec PRIVATE nncc_common)
target_link_libraries(pepper_csv2vec PUBLIC nncc_coverage)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Google Test is mandatory for test
nnas_find_package(GTest REQUIRED)

GTest_AddTest(pepper_csv2vec_test ${TESTS})
target_link_libraries(pepper_csv2vec_test pepper_csv2vec)
