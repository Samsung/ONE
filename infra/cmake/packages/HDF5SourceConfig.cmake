function(_HDF5Source_import)
  if(NOT DOWNLOAD_HDF5)
    set(HDF5Source_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_HDF5)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(HDF5_URL https://github.com/HDFGroup/hdf5/archive/hdf5-1_8_16.tar.gz)

  ExternalSource_Download(HDF5 ${HDF5_URL})

  set(HDF5Source_DIR ${HDF5_SOURCE_DIR} PARENT_SCOPE)
  set(HDF5Source_FOUND TRUE PARENT_SCOPE)
endfunction(_HDF5Source_import)

_HDF5Source_import()
