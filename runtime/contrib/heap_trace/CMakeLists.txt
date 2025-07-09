if(NOT BUILD_HEAP_TRACE)
  return()
endif(NOT BUILD_HEAP_TRACE)

add_library(heap_trace SHARED src/cl_create_buffer_stub.cc
                              src/cl_release_mem_object.cc
                              src/cl_retain_mem_object_stub.cc
                              src/free_stub.cc
                              src/malloc_stub.cc
                              src/realloc_stub.cc
                              src/valloc_stub.cc
                              src/calloc_stub.cc
                              src/posix_memalign_stub.cc
                              src/aligned_alloc_stub.cc
                              src/symbol_searcher.cc
                              src/trace.cc
                              src/memory_pool_for_symbol_searcher_internals.cc
)
target_link_libraries(heap_trace PRIVATE ${CMAKE_DL_LIBS})

add_subdirectory(tests)

install(TARGETS heap_trace DESTINATION ${CMAKE_INSTALL_LIBDIR})
