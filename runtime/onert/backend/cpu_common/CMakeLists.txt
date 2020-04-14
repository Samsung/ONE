file(GLOB SOURCES "*.cc")
file(GLOB_RECURSE TESTS "*.test.cc")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(${LIB_ONERT_BACKEND_CPU_COMMON} STATIC ${SOURCES})

target_include_directories(${LIB_ONERT_BACKEND_CPU_COMMON} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${LIB_ONERT_BACKEND_CPU_COMMON} PUBLIC onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_CPU_COMMON} PRIVATE nnfw_lib_misc)
target_link_libraries(${LIB_ONERT_BACKEND_CPU_COMMON} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_CPU_COMMON} PRIVATE nnfw_coverage)

set_target_properties(${LIB_ONERT_BACKEND_CPU_COMMON} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${LIB_ONERT_BACKEND_CPU_COMMON} PROPERTIES OUTPUT_NAME backend_cpu_common)

install(TARGETS ${LIB_ONERT_BACKEND_CPU_COMMON} ARCHIVE DESTINATION lib)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        DESTINATION "include/onert"
        FILES_MATCHING PATTERN "*.h"
        )

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Unit Tests
set(TEST_ONERT_BACKEND_CPU_COMMON test_onert_backend_cpu_common)

add_executable(${TEST_ONERT_BACKEND_CPU_COMMON} ${TESTS})

target_link_libraries(${TEST_ONERT_BACKEND_CPU_COMMON} ${LIB_ONERT_BACKEND_CPU_COMMON})
target_link_libraries(${TEST_ONERT_BACKEND_CPU_COMMON} gtest gtest_main dl ${LIB_PTHREAD})

add_test(${TEST_ONERT_BACKEND_CPU_COMMON} ${TEST_ONERT_BACKEND_CPU_COMMON})
install(TARGETS ${TEST_ONERT_BACKEND_CPU_COMMON} DESTINATION unittest)
