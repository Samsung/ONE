set(TESTS_OPTIMIZATIONS_SRC CombineTransposes.cpp SinkTest.cpp DeadCodeElimination.cpp FuseArithmeticOps.cpp)
nnc_add_unit_test(tests_for_optimizations ${TESTS} ${TESTS_OPTIMIZATIONS_SRC})
optional_target_link_libraries(tests_for_optimizations nnc_optimizations mir)
