set(CL_SOURCES ${NNC_SUPPORT_DIR}/CommandLine.cpp)
set(CL_TESTS CommandLineTest.cpp)

nnc_add_unit_test(nnc_support_commandline_test ${CL_TESTS} ${CL_SOURCES})
