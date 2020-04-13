function(_TensorFlowLite_import)
  nnas_find_package(TensorFlowSource EXACT 1.7 QUIET)

  if(NOT TensorFlowSource_FOUND)
    set(TensorFlowLite_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT TensorFlowSource_FOUND)

  # TensorFlow 1.7 downloads FlatBuffers from https://github.com/google/flatbuffers/archive/971a68110e4.tar.gz
  #
  # FlatBuffers 1.8 is compatible with 971a68110e4.
  nnas_find_package(FlatBuffersSource EXACT 1.8 QUIET)

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

  if(NOT TARGET tensorflowlite-1.7)
    nnas_include(ExternalProjectTools)
    add_extdirectory("${CMAKE_CURRENT_LIST_DIR}/Lite" tflite-1.7)
  endif(NOT TARGET tensorflowlite-1.7)

  set(TensorFlowLite_FOUND TRUE PARENT_SCOPE)
endfunction(_TensorFlowLite_import)

_TensorFlowLite_import()
