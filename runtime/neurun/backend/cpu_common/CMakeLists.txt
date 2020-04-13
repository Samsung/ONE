file(GLOB SOURCES "*.cc")
file(GLOB_RECURSE TESTS "*.test.cc")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(${LIB_NEURUN_BACKEND_CPU_COMMON} STATIC ${SOURCES})

target_include_directories(${LIB_NEURUN_BACKEND_CPU_COMMON} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${LIB_NEURUN_BACKEND_CPU_COMMON} PUBLIC neurun_core)
target_link_libraries(${LIB_NEURUN_BACKEND_CPU_COMMON} PRIVATE nnfw_lib_misc)
target_link_libraries(${LIB_NEURUN_BACKEND_CPU_COMMON} PRIVATE nnfw_common)
target_link_libraries(${LIB_NEURUN_BACKEND_CPU_COMMON} PRIVATE nnfw_coverage)

set_target_properties(${LIB_NEURUN_BACKEND_CPU_COMMON} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${LIB_NEURUN_BACKEND_CPU_COMMON} PROPERTIES OUTPUT_NAME backend_cpu_common)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Unit Tests
set(TEST_NEURUN_BACKEND_CPU_COMMON test_neurun_backend_cpu_common)

add_executable(${TEST_NEURUN_BACKEND_CPU_COMMON} ${TESTS})

target_link_libraries(${TEST_NEURUN_BACKEND_CPU_COMMON} ${LIB_NEURUN_BACKEND_CPU_COMMON})
target_link_libraries(${TEST_NEURUN_BACKEND_CPU_COMMON} gtest gtest_main dl ${LIB_PTHREAD})

add_test(${TEST_NEURUN_BACKEND_CPU_COMMON} ${TEST_NEURUN_BACKEND_CPU_COMMON})
install(TARGETS ${TEST_NEURUN_BACKEND_CPU_COMMON} DESTINATION unittest)
