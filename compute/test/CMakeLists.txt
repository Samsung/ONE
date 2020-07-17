set(TEST_COMPUTE test_compute)

file(GLOB_RECURSE TESTS "*.cc")

add_executable(${TEST_COMPUTE} ${TESTS})

target_link_libraries(${TEST_COMPUTE} nnfw_lib_cker)
target_link_libraries(${TEST_COMPUTE} gtest)
target_link_libraries(${TEST_COMPUTE} gtest_main)
target_link_libraries(${TEST_COMPUTE} ${LIB_PTHREAD} dl)
add_test(${TEST_COMPUTE} ${TEST_COMPUTE})

install(TARGETS ${TEST_COMPUTE} DESTINATION unittest_standalone)
