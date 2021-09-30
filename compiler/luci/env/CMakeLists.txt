file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(luci_env ${LIBRARY_TYPE} ${SOURCES})
target_include_directories(luci_env PUBLIC include)
target_link_libraries(luci_env PRIVATE nncc_common)
install(TARGETS luci_env DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(luci_env_test ${TESTS})
target_include_directories(luci_env_test PRIVATE src)
target_link_libraries(luci_env_test luci_env)
