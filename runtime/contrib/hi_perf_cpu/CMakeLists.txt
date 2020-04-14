set(LIB_ONERT_BACKEND_HI_PERF_CPU onert_backend_hi_perf)

nnas_find_package(NNPACK QUIET)

option(BUILD_ONERT_HI_PERF_CPU_BACKEND
  "Build onert HI_PERF_CPU backend"
  ${NNPACK_FOUND} # Default value when there is no explicit user request
)

message(STATUS "Build onert HI_PERF_CPU backend: ${BUILD_ONERT_HI_PERF_CPU_BACKEND}")

if(NOT BUILD_ONERT_HI_PERF_CPU_BACKEND)
  return()
endif(NOT BUILD_ONERT_HI_PERF_CPU_BACKEND)

file(GLOB_RECURSE SOURCES "*.cc")
file(GLOB_RECURSE TESTS "*.test.cc")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(${LIB_ONERT_BACKEND_HI_PERF_CPU} SHARED ${SOURCES})

target_link_libraries(${LIB_ONERT_BACKEND_HI_PERF_CPU} PRIVATE nnfw_lib_misc)
target_link_libraries(${LIB_ONERT_BACKEND_HI_PERF_CPU} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_HI_PERF_CPU} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_HI_PERF_CPU} PRIVATE nnpack pthreadpool cpuinfo)
target_link_libraries(${LIB_ONERT_BACKEND_HI_PERF_CPU} PRIVATE nnfw_coverage)
target_include_directories(${LIB_ONERT_BACKEND_HI_PERF_CPU} PRIVATE ${NNPACK_INCLUDE_DIRS})

set_target_properties(${LIB_ONERT_BACKEND_HI_PERF_CPU} PROPERTIES OUTPUT_NAME backend_NNPACK)

install(TARGETS ${LIB_ONERT_BACKEND_HI_PERF_CPU} DESTINATION lib)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Unit Tests
set(TEST_ONERT_BACKEND_HI_PERF_CPU test_onert_backend_hi_perf)

add_executable(${TEST_ONERT_BACKEND_HI_PERF_CPU} ${TESTS})

target_link_libraries(${TEST_ONERT_BACKEND_HI_PERF_CPU} ${LIB_ONERT_BACKEND_HI_PERF_CPU})
target_link_libraries(${TEST_ONERT_BACKEND_HI_PERF_CPU} gtest gtest_main ${LIB_PTHREAD})
target_link_libraries(${TEST_ONERT_BACKEND_HI_PERF_CPU} nnpack)

add_test(${TEST_ONERT_BACKEND_HI_PERF_CPU} ${TEST_ONERT_BACKEND_HI_PERF_CPU})
install(TARGETS ${TEST_ONERT_BACKEND_HI_PERF_CPU} DESTINATION unittest)
