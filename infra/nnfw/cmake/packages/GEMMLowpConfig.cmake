function(_GEMMLowp_import)
  nnas_find_package(TensorFlowGEMMLowpSource EXACT 2.16.1 QUIET)

  if(NOT TensorFlowGEMMLowpSource_FOUND)
    set(GEMMLowp_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT TensorFlowGEMMLowpSource_FOUND)

  if(NOT TARGET gemmlowp)
    find_package(Threads REQUIRED)

    add_library(gemmlowp INTERFACE)
    target_include_directories(gemmlowp SYSTEM INTERFACE ${TensorFlowGEMMLowpSource_DIR})
    target_link_libraries(gemmlowp INTERFACE ${LIB_PTHREAD})
  endif(NOT TARGET gemmlowp)

  set(GEMMLowp_FOUND TRUE PARENT_SCOPE)
endfunction(_GEMMLowp_import)

_GEMMLowp_import()
