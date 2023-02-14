function(_HDF5Source_import)
  if(NOT DOWNLOAD_HDF5)
    set(HDF5Source_FOUND
        FALSE
        PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_HDF5)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(HDF5_URL ${EXTERNAL_DOWNLOAD_SERVER}/HDFGroup/hdf5/archive/hdf5-1_8_16.tar.gz)

  ExternalSource_Download(HDF5 ${HDF5_URL} PATCH ${CMAKE_CURRENT_LIST_DIR}/HDF5Source.patch)

  set(HDF5Source_DIR
      ${HDF5_SOURCE_DIR}
      PARENT_SCOPE)
  set(HDF5Source_FOUND
      TRUE
      PARENT_SCOPE)
endfunction(_HDF5Source_import)

_HDF5Source_import()
