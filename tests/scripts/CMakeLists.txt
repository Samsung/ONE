if(NOT INSTALL_TEST_SCRIPTS)
  return()
endif(NOT INSTALL_TEST_SCRIPTS)

# Install test scripts
set(TEST_SCRIPTS
  "benchmark_nnpkg.sh"
  "benchmark_ops.sh"
  "merge_result_of_benchmark_nnpkg.py"
  "print_to_json.sh"
  "test_ubuntu_runtime.sh"
  "test_ubuntu_runtime_mixed.sh"
)

if(ANDROID)
  list(APPEND TEST_SCRIPTS "common_android.sh" "test_scheduler_with_profiling_android.sh")
else(ANDROID)
  list(APPEND TEST_SCRIPTS "common.sh" "test_scheduler_with_profiling.sh")
endif(ANDROID)

if("${TARGET_ARCH_BASE}" STREQUAL "arm")
  list(APPEND TEST_SCRIPTS "test_arm_nnpkg.sh")
endif("${TARGET_ARCH_BASE}" STREQUAL "arm")

if(ENABLE_COVERAGE)
  list(APPEND TEST_SCRIPTS "test_coverage.sh")
endif(ENABLE_COVERAGE)

install(PROGRAMS ${TEST_SCRIPTS} DESTINATION test/scripts)

# Install test list
file(GLOB TEST_LIST_DIR list)
install(DIRECTORY ${TEST_LIST_DIR} DESTINATION test)
