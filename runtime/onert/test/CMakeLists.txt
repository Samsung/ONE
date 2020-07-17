set(TEST_ONERT test_onert)

file(GLOB_RECURSE TESTS "*.cc")

add_executable(${TEST_ONERT} ${TESTS})

target_include_directories(${TEST_ONERT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../core/src)

target_link_libraries(${TEST_ONERT} onert_core)
target_link_libraries(${TEST_ONERT} gtest)
target_link_libraries(${TEST_ONERT} gtest_main)
target_link_libraries(${TEST_ONERT} ${LIB_PTHREAD} dl)
add_test(${TEST_ONERT} ${TEST_ONERT})

install(TARGETS ${TEST_ONERT} DESTINATION unittest_standalone)
