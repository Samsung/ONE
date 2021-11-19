file(GLOB_RECURSE HEADERS "include/*.h")
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

# NOTE STATIC is deliberately used here to allow clients to use 'angkor' without installation
add_library(angkor STATIC ${HEADERS} ${SOURCES})
if (NOT ANGKOR_NO_PIC)
  set_target_properties(angkor PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif (NOT ANGKOR_NO_PIC)
set_target_properties(angkor PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(angkor PUBLIC include)
target_link_libraries(angkor PRIVATE nncc_common)
target_link_libraries(angkor PUBLIC nncc_coverage)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Google Test is mandatory for test
nnas_find_package(GTest REQUIRED)

GTest_AddTest(angkor_test ${TESTS})
target_link_libraries(angkor_test angkor)
