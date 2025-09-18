add_library(ndarray STATIC src/Array.cpp src/ContiguousSpan.cpp)

set_target_properties(ndarray PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(ndarray PUBLIC include)

option(NDARRAY_INLINE_TEMPLATES "Set to ON to disable extern declarations for common types")

if(${NDARRAY_INLINE_TEMPLATES})
    target_compile_definitions(ndarray PUBLIC -DNDARRAY_INLINE_TEMPLATES=1)
endif()

target_link_libraries(ndarray PRIVATE nnfw_common)
target_link_libraries(ndarray PRIVATE nnfw_coverage)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

add_executable(ndarray_test src/Array.test.cpp src/ContiguousSpan.test.cpp)
target_link_libraries(ndarray_test PRIVATE ndarray)
target_link_libraries(ndarray_test PRIVATE nnfw_common nnfw_coverage)
target_link_libraries(ndarray_test PUBLIC gtest gtest_main Threads::Threads)

add_test(ndarray_test ndarray_test)
install(TARGETS ndarray_test DESTINATION unittest)

add_subdirectory(example)
