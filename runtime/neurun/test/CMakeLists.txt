set(TEST_NEURUN test_neurun)

file(GLOB_RECURSE TESTS "*.cc")

add_executable(${TEST_NEURUN} ${TESTS})

target_include_directories(${TEST_NEURUN} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../core/src)

target_link_libraries(${TEST_NEURUN} neurun_core)
target_link_libraries(${TEST_NEURUN} gtest)
target_link_libraries(${TEST_NEURUN} gtest_main)
target_link_libraries(${TEST_NEURUN} ${LIB_PTHREAD} dl)
add_test(${TEST_NEURUN} ${TEST_NEURUN})

install(TARGETS ${TEST_NEURUN} DESTINATION unittest)
