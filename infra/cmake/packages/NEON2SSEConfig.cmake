function(_NEON2SSE_import)
  nnas_find_package(NEON2SSESource QUIET)

  if(NOT NEON2SSESource_FOUND)
    set(NEON2SSE_FOUND
        FALSE
        PARENT_SCOPE)
    return()
  endif(NOT NEON2SSESource_FOUND)

  if(NOT TARGET neon2sse)
    add_library(neon2sse INTERFACE)
    target_include_directories(neon2sse INTERFACE "${NEON2SSESource_DIR}")
  endif(NOT TARGET neon2sse)

  set(NEON2SSE_FOUND
      TRUE
      PARENT_SCOPE)
endfunction(_NEON2SSE_import)

_NEON2SSE_import()
