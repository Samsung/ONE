if(NOT TARGET mio_tf)
  return()
endif(NOT TARGET mio_tf)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(plier_tf STATIC ${SOURCES})
set_target_properties(plier_tf PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(plier_tf PUBLIC include)

target_link_libraries(plier_tf PUBLIC angkor)
target_link_libraries(plier_tf PUBLIC loco)
target_link_libraries(plier_tf PUBLIC mio_tf)

# Apply global configurations (e.g. warnings as error)
target_link_libraries(plier_tf PRIVATE nncc_common)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Google Test is mandatory for test
nnas_find_package(GTest REQUIRED)

GTest_AddTest(plier_tf_test ${TESTS})
target_link_libraries(plier_tf_test plier_tf)
