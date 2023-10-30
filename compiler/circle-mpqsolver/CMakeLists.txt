nnas_find_package(Jsoncpp)
if(NOT Jsoncpp_FOUND)
    message(STATUS "Build circle-mpqsolver: FAILED (missing jsoncpp)")
    return()
endif(NOT Jsoncpp_FOUND)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_executable(circle-mpqsolver "${SOURCES}")
target_include_directories(circle-mpqsolver PRIVATE src)
target_include_directories(circle-mpqsolver PRIVATE ${Jsoncpp_INCLUDE_DIRS})
target_link_libraries(circle-mpqsolver ${Jsoncpp_STATIC_LIB})
target_link_libraries(circle-mpqsolver arser)
target_link_libraries(circle-mpqsolver vconone)
target_link_libraries(circle-mpqsolver safemain)
target_link_libraries(circle-mpqsolver luci_service)
target_link_libraries(circle-mpqsolver luci_pass)
target_link_libraries(circle-mpqsolver luci_interpreter)
target_link_libraries(circle-mpqsolver dio_hdf5)
target_link_libraries(circle-mpqsolver luci_import)
target_link_libraries(circle-mpqsolver luci_export)
target_link_libraries(circle-mpqsolver luci_log)
target_link_libraries(circle-mpqsolver nncc_common)

install(TARGETS circle-mpqsolver DESTINATION bin)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# circle-mpqsolver is executable, so we do not link it to the test.
# Instead, we use TEST_SOURCES to specify sources uesd for tests.
set(TEST_SOURCES
    "src/bisection/DepthParameterizer.cpp"
    "src/core/Quantizer.cpp"
    "src/bisection/VISQErrorApproximator.cpp"
    "src/core/ErrorMetric.cpp"
    "src/pattern/PatternResolver.cpp"
    "src/pattern/PatternSolver.cpp"
    "src/core/Dumper.cpp"
    "src/core/DumpingHooks.cpp"
    "src/core/Evaluator.cpp"
    "src/MPQSolver.cpp"
    "src/core/SolverOutput.cpp"
    "src/bisection/BisectionSolver.cpp"
)

nnas_find_package(GTest REQUIRED)
GTest_AddTest(circle_mpqsolver_test ${TESTS} ${TEST_SOURCES})
target_include_directories(circle_mpqsolver_test PRIVATE src)
target_include_directories(circle_mpqsolver_test PRIVATE ${Jsoncpp_INCLUDE_DIRS})
target_link_libraries(circle_mpqsolver_test ${Jsoncpp_STATIC_LIB})
target_link_libraries(circle_mpqsolver_test luci_service)
target_link_libraries(circle_mpqsolver_test luci_pass)
target_link_libraries(circle_mpqsolver_test luci_testhelper)
target_link_libraries(circle_mpqsolver_test luci_import)
target_link_libraries(circle_mpqsolver_test luci_export)
target_link_libraries(circle_mpqsolver_test luci_interpreter)
target_link_libraries(circle_mpqsolver_test dio_hdf5)
