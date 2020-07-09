function(_TensorFlowLite_import)
  nnas_find_package(TensorFlowSource EXACT 1.13.1 QUIET)

  if(NOT TensorFlowSource_FOUND)
    set(TensorFlowLite_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT TensorFlowSource_FOUND)

  nnas_find_package(FlatBuffersSource EXACT 1.10 QUIET)

  if(NOT FlatBuffersSource_FOUND)
    set(TensorFlowLite_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT FlatBuffersSource_FOUND)

  nnas_find_package(Farmhash QUIET)

  if(NOT Farmhash_FOUND)
    set(TensorFlowLite_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT Farmhash_FOUND)

  nnas_find_package(Eigen QUIET)

  if(NOT Eigen_FOUND)
    set(TensorFlowLite_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT Eigen_FOUND)

  nnas_find_package(GEMMLowp QUIET)

  if(NOT GEMMLowp_FOUND)
    set(TensorFlowLite_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT GEMMLowp_FOUND)

  nnas_find_package(NEON2SSE QUIET)

  if(NOT NEON2SSE_FOUND)
    set(TensorFlowLite_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT NEON2SSE_FOUND)

  nnas_find_package(Abseil QUIET)

  if(NOT Abseil_FOUND)
    set(TensorFlowLite_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT Abseil_FOUND)

  if(NOT TARGET tensorflowlite-1.13.1)
    nnas_include(ExternalProjectTools)
    add_extdirectory("${CMAKE_CURRENT_LIST_DIR}/Lite" tflite-1.13.1)
  endif(NOT TARGET tensorflowlite-1.13.1)

  set(TensorFlowLite_FOUND TRUE PARENT_SCOPE)
endfunction(_TensorFlowLite_import)

_TensorFlowLite_import()
