set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_executable(circle-eval-diff ${DRIVER} ${SOURCES})
target_include_directories(circle-eval-diff PRIVATE include)

target_link_libraries(circle-eval-diff arser)
target_link_libraries(circle-eval-diff safemain)
target_link_libraries(circle-eval-diff foder)
target_link_libraries(circle-eval-diff loco)
target_link_libraries(circle-eval-diff luci_import)
target_link_libraries(circle-eval-diff luci_lang)
target_link_libraries(circle-eval-diff luci_interpreter)
target_link_libraries(circle-eval-diff dio_hdf5)
target_link_libraries(circle-eval-diff vconone)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# circle-eval-diff is executable, so we do not link it to the test.
# Instead, we use TEST_SOURCES to specify sources uesd for tests.
set(TEST_SOURCES
    "src/MetricPrinter.cpp"
    "src/Tensor.cpp")

nnas_find_package(GTest REQUIRED)
GTest_AddTest(circle_eval_diff_test ${TESTS} ${TEST_SOURCES})
target_include_directories(circle_eval_diff_test PRIVATE src)
target_link_libraries(circle_eval_diff_test luci_testhelper)
target_link_libraries(circle_eval_diff_test nncc_coverage)
