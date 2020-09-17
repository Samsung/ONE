function(_GEMMLowp_import)
  nnas_find_package(GEMMLowpSource QUIET)

  if(NOT GEMMLowpSource_FOUND)
    set(GEMMLowp_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT GEMMLowpSource_FOUND)

  if(NOT TARGET gemmlowp)
    find_package(Threads REQUIRED)

    add_library(gemmlowp INTERFACE)
    target_include_directories(gemmlowp SYSTEM INTERFACE ${GEMMLowpSource_DIR})
    target_link_libraries(gemmlowp INTERFACE ${LIB_PTHREAD})
  endif(NOT TARGET gemmlowp)

  set(GEMMLowp_FOUND TRUE PARENT_SCOPE)
endfunction(_GEMMLowp_import)

_GEMMLowp_import()
