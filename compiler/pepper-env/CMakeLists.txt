file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(pepper_env STATIC ${SOURCES})
#set_target_properties(pepper_env PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pepper_env PUBLIC include)
target_link_libraries(pepper_env PRIVATE nncc_common)
target_link_libraries(pepper_env PUBLIC nncc_coverage)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Google Test is mandatory for test
nnas_find_package(GTest REQUIRED)

GTest_AddTest(pepper_env_test ${TESTS})
target_link_libraries(pepper_env_test pepper_env)
