file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(luci_profile ${LIBRARY_TYPE} ${SOURCES})
target_include_directories(luci_profile PRIVATE src)
target_include_directories(luci_profile PUBLIC include)
target_link_libraries(luci_profile PUBLIC loco)
target_link_libraries(luci_profile PUBLIC luci_lang)

install(TARGETS luci_profile DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(luci_profile_test ${TESTS})
target_include_directories(luci_profile_test PRIVATE src)
target_link_libraries(luci_profile_test luci_lang)
target_link_libraries(luci_profile_test luci_profile)
