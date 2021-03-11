file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(loco SHARED ${SOURCES})
target_include_directories(loco PUBLIC include)
# TODO Remove dependencies on angkor library
target_link_libraries(loco PUBLIC angkor)
target_link_libraries(loco PRIVATE stdex)
# Let's apply nncc common compile options
#
# NOTE This will enable strict compilation (warnings as error).
#      Please refer to the top-level CMakeLists.txt for details
if(NOT LUCI_MICRO)
	target_link_libraries(loco PRIVATE nncc_common)
	target_link_libraries(loco PUBLIC nncc_coverage)
endif(NOT LUCI_MICRO)
# Q. HOW TO MAKE DEV PACKAGE(?)
install(TARGETS loco DESTINATION lib)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Google Test is mandatory for internal testing
nnas_find_package(GTest REQUIRED)

GTest_AddTest(loco_test ${TESTS})
target_link_libraries(loco_test stdex)
target_link_libraries(loco_test loco)
