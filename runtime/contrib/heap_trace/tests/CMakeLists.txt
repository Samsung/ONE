set(HEAP_TRACE_TESTS heap_trace_test)

find_package(OpenCL REQUIRED)

add_library(test_sample1 SHARED src/test_sample1/test_sample1.cc)

add_library(test_sample2 SHARED src/test_sample2/test_sample2.cc)
target_link_libraries(test_sample2 test_sample3)

add_library(test_sample3 SHARED src/test_sample3/test_sample3.cc)

add_library(test_sample4 SHARED src/test_sample4/test_sample4.cc)

add_executable(${HEAP_TRACE_TESTS} src/cl_release_mem_object_interception_test.cc
        src/symbol_searcher_test.cc
        src/cl_create_buffer_interception_test.cc
        src/cl_retain_mem_object_interception_test.cc
        src/free_interception_test.cc
        src/main.cc
        src/common_test_environment.cc
        src/trace_test.cc
        src/malloc_interception_test.cc
        src/realloc_interception_test.cc
        src/valloc_interception_test.cc
        src/calloc_interception_test.cc
        src/posix_memalign_interception_test.cc
        src/aligned_alloc_interception_test.cc
        src/file_content_manipulations.cc
        src/memory_pool_for_symbol_searcher_internals_test.cc
        )
target_include_directories(${HEAP_TRACE_TESTS} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(${HEAP_TRACE_TESTS} test_sample1)
target_link_libraries(${HEAP_TRACE_TESTS} heap_trace)
target_link_libraries(${HEAP_TRACE_TESTS} test_sample4)
target_link_libraries(${HEAP_TRACE_TESTS} gtest)
target_link_libraries(${HEAP_TRACE_TESTS} gtest_main)
target_link_libraries(${HEAP_TRACE_TESTS} Threads::Threads)
target_link_libraries(${HEAP_TRACE_TESTS} ${CMAKE_DL_LIBS})
target_link_libraries(${HEAP_TRACE_TESTS} ${OpenCL_LIBRARY})
target_link_libraries(${HEAP_TRACE_TESTS} stdc++fs)

add_test(${HEAP_TRACE_TESTS} ${HEAP_TRACE_TESTS})

install(TARGETS test_sample1 DESTINATION unittest)
install(TARGETS test_sample2 DESTINATION unittest)
install(TARGETS test_sample3 DESTINATION unittest)
install(TARGETS test_sample4 DESTINATION unittest)
install(TARGETS ${HEAP_TRACE_TESTS} DESTINATION unittest)
