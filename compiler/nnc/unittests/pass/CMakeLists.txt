file(GLOB_RECURSE TEST_SOURCES "*.cpp")

nnc_add_unit_test(nnc_pass_test ${TEST_SOURCES})
optional_target_link_libraries(nnc_pass_test mir)
