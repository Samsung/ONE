file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(locoex_customop SHARED ${SOURCES})
target_include_directories(locoex_customop PUBLIC include)
target_link_libraries(locoex_customop PUBLIC loco)
target_link_libraries(locoex_customop PRIVATE locop pepper_str)
install(TARGETS locoex_customop DESTINATION lib)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(locoex_customop_test ${TESTS})
target_link_libraries(locoex_customop_test loco locoex_customop)
