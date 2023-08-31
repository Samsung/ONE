# Luci library is not supported is on cross build
if(CMAKE_CROSSCOMPILING)
  return()
endif()

nnfw_find_package(Luci QUIET)
if(NOT Luci_FOUND)
  message(STATUS "Luci not found. Skip onert_odc")
  return()
endif()

file(GLOB_RECURSE SOURCES "*.cc")
file(GLOB_RECURSE TESTS "*.test.cc")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(onert_odc SHARED ${SOURCES})
target_link_libraries(onert_odc PRIVATE onert_core luci::import luci::export luci::pass luci::loco)
target_link_libraries(onert_odc PRIVATE nnfw_common)
target_link_libraries(onert_odc PRIVATE nnfw_coverage)

install(TARGETS onert_odc LIBRARY DESTINATION lib/nnfw/odc)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Unit Tests
set(TEST_ONERT_ODC test_onert_odc)

add_executable(${TEST_ONERT_ODC} ${TESTS})

target_link_libraries(${TEST_ONERT_ODC} onert_odc)
# Requires linking nnfw_coverage: check header coverage
target_link_libraries(${TEST_ONERT_CORE} nnfw_coverage)
target_link_libraries(${TEST_ONERT_ODC} gtest gtest_main dl ${LIB_PTHREAD})
target_include_directories(${TEST_ONERT_ODC} PRIVATE $<TARGET_PROPERTY:onert_odc,INCLUDE_DIRECTORIES>)

add_test(${TEST_ONERT_ODC} ${TEST_ONERT_ODC})
install(TARGETS ${TEST_ONERT_ODC} DESTINATION unittest)
