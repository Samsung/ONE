add_library(ndarray STATIC src/Array.cpp src/ContiguousSpan.cpp)

set_target_properties(ndarray PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(ndarray PUBLIC include)
#can't make this private because of c++ templates
target_include_directories(ndarray PUBLIC src)

option(NDARRAY_INLINE_TEMPLATES "Set to ON to disable extern declarations for common types")

if(${NDARRAY_INLINE_TEMPLATES})
    target_compile_definitions(ndarray PUBLIC -DNDARRAY_INLINE_TEMPLATES=1)
endif()

target_link_libraries(ndarray PRIVATE nnfw_common)
target_link_libraries(ndarray PRIVATE nnfw_coverage)
if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

add_subdirectory(test)
add_subdirectory(example)
