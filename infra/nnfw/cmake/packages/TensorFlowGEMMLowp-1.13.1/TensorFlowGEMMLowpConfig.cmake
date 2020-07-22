function(_GEMMLowp_import)
  nnas_find_package(GEMMLowpSource QUIET)

  if(NOT GEMMLowpSource_FOUND)
    set(TensorFlowGEMMLowp_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT GEMMLowpSource_FOUND)

  if(NOT TARGET gemmlowp-tf-1.13.1)
    find_package(Threads REQUIRED)

    add_library(gemmlowp-tf-1.13.1 INTERFACE)
    target_include_directories(gemmlowp-tf-1.13.1 SYSTEM INTERFACE ${GEMMLowpSource_DIR})
    target_link_libraries(gemmlowp-tf-1.13.1 INTERFACE ${LIB_PTHREAD})
  endif(NOT TARGET gemmlowp-tf-1.13.1)

  set(TensorFlowGEMMLowp_FOUND TRUE PARENT_SCOPE)
endfunction(_GEMMLowp_import)

_GEMMLowp_import()
