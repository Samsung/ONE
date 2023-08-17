set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(record-minmax ${DRIVER} ${SOURCES})
target_include_directories(record-minmax PRIVATE include)

target_link_libraries(record-minmax arser)
target_link_libraries(record-minmax safemain)
target_link_libraries(record-minmax luci_import)
target_link_libraries(record-minmax luci_env)
target_link_libraries(record-minmax luci_export)
target_link_libraries(record-minmax luci_interpreter)
target_link_libraries(record-minmax luci_log)
target_link_libraries(record-minmax dio_hdf5)
target_link_libraries(record-minmax vconone)
target_link_libraries(record-minmax nncc_coverage)
target_link_libraries(record-minmax nncc_common)

install(TARGETS record-minmax DESTINATION bin)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

### 
### record-minmax-for-thread-test is disabled, because gcc package has a bug.
### (https://bugs.launchpad.net/ubuntu/+source/gcc-10/+bug/2029910)
### Let's enable the target after the bug is fixed.
###
# Build record-minmax-for-thread-test if target arch is 64bit
# Thread sanitizer is only available on 64bit machine
# (https://github.com/google/sanitizers/wiki/ThreadSanitizerCppManual#supported-platforms)
# if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "8")
#   # create record-minmax-for-thread-test target
#   # Note: record-minmax-for-thread-test is built with -fsanitize=thread so that thread sanitizer can check memory bugs,
#   # record-minmax is built without the option for performance.
#   add_executable(record-minmax-for-thread-test ${DRIVER} ${SOURCES})
#   target_include_directories(record-minmax-for-thread-test PRIVATE include)

#   target_link_libraries(record-minmax-for-thread-test arser)
#   target_link_libraries(record-minmax-for-thread-test safemain)
#   target_link_libraries(record-minmax-for-thread-test luci_import)
#   target_link_libraries(record-minmax-for-thread-test luci_env)
#   target_link_libraries(record-minmax-for-thread-test luci_export)
#   target_link_libraries(record-minmax-for-thread-test luci_interpreter)
#   target_link_libraries(record-minmax-for-thread-test dio_hdf5)
#   target_link_libraries(record-minmax-for-thread-test vconone)
#   target_link_libraries(record-minmax-for-thread-test nncc_coverage)
#   target_link_libraries(record-minmax-for-thread-test luci_log)

#   target_compile_options(record-minmax-for-thread-test PUBLIC -fsanitize=thread)
#   target_link_libraries(record-minmax-for-thread-test -fsanitize=thread)
# endif("${CMAKE_SIZEOF_VOID_P}" STREQUAL "8")

file(GLOB_RECURSE TESTS "tests/*.test.cpp")

nnas_find_package(GTest REQUIRED)
GTest_AddTest(record_minmax_function_test "${TESTS}")
target_include_directories(record_minmax_function_test PRIVATE include)
target_link_libraries(record_minmax_function_test nncc_coverage)
