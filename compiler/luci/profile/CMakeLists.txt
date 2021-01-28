file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
#list(REMOVE_ITEM SOURCES ${TESTS})

add_library(luci_profile SHARED ${SOURCES})
target_include_directories(luci_profile PRIVATE src)
target_include_directories(luci_profile PUBLIC include)
target_link_libraries(luci_profile PUBLIC loco)

install(TARGETS luci_profile DESTINATION lib)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# nnas_find_package(GTest REQUIRED)

# GTest_AddTest(luci_profile_test ${TESTS})
# target_include_directories(luci_profile_test PRIVATE src)
