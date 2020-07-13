function(_Caffe_import)
  nnas_find_package(CaffeSource QUIET)

  if(NOT CaffeSource_FOUND)
    set(Caffe_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT CaffeSource_FOUND)

  nnas_find_package(CaffeProto QUIET)

  if(NOT CaffeProto_FOUND)
    set(Caffe_FOUND FALSE PARENT_SCOPE)
    return()
  endif()

  find_package(Boost 1.54 COMPONENTS system thread filesystem QUIET)

  if(NOT Boost_FOUND)
    set(Caffe_FOUND FALSE PARENT_SCOPE)
    return()
  endif()

  # TODO This will be nnas_find_package
  find_package(HDF5 COMPONENTS HL QUIET CONFIG)

  if(NOT HDF5_FOUND)
    find_package(HDF5 COMPONENTS HL QUIET MODULE)
  endif(NOT HDF5_FOUND)

  if(NOT HDF5_FOUND)
    set(Caffe_FOUND FALSE PARENT_SCOPE)
    return()
  endif()

  list(APPEND CMAKE_MODULE_PATH ${CaffeSource_DIR}/cmake/Modules)

  find_package(Atlas QUIET)

  if(NOT ATLAS_FOUND)
    set(Caffe_FOUND FALSE PARENT_SCOPE)
    return()
  endif()

  nnas_find_package(GLog QUIET)

  if(NOT GLog_FOUND)
    set(Caffe_FOUND FALSE PARENT_SCOPE)
    return()
  endif()

  nnas_find_package(GFlags QUIET)

  if(NOT GFlags_FOUND)
    set(Caffe_FOUND FALSE PARENT_SCOPE)
    return()
  endif()

  if(NOT TARGET caffe)
    nnas_include(ExternalProjectTools)
    add_extdirectory("${CMAKE_CURRENT_LIST_DIR}/Caffe" caffe)
    message(STATUS "Found Caffe: TRUE")
  endif(NOT TARGET caffe)

  set(Caffe_FOUND TRUE PARENT_SCOPE)
endfunction(_Caffe_import)

_Caffe_import()
