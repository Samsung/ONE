set(TESTS_TRANSFORMATIONS_SRC Switcher.cpp)

nnc_add_unit_test(nnc_transformations_test ${TESTS} ${TESTS_TRANSFORMATIONS_SRC})
optional_target_link_libraries(nnc_transformations_test nnc_transformations mir)
