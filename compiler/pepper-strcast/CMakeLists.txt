file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(pepper_strcast STATIC ${SOURCES})
if (NOT PEPPER_STRCAST_NO_PIC)
  set_target_properties(pepper_strcast PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif(NOT PEPPER_STRCAST_NO_PIC)
target_include_directories(pepper_strcast PUBLIC include)
target_link_libraries(pepper_strcast PRIVATE nncc_common)
target_link_libraries(pepper_strcast PUBLIC nncc_coverage)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Google Test is mandatory for test
nnas_find_package(GTest REQUIRED)

GTest_AddTest(pepper_strcast_test ${TESTS})
target_link_libraries(pepper_strcast_test pepper_strcast)
