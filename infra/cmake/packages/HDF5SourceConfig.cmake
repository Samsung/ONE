function(_HDF5Source_import)
  if(NOT DOWNLOAD_HDF5)
    set(HDF5Source_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_HDF5)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(HDF5_URL https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.16/src/hdf5-1.8.16.tar.gz)

  ExternalSource_Download(HDF5 ${HDF5_URL})

  set(HDF5Source_DIR ${HDF5_SOURCE_DIR} PARENT_SCOPE)
  set(HDF5Source_FOUND TRUE PARENT_SCOPE)
endfunction(_HDF5Source_import)

_HDF5Source_import()
