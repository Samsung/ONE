function(_LibnpySource_import)
  if(NOT ${DOWNLOAD_LIBNPY})
    set(LibnpySource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${DOWNLOAD_LIBNPY})

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(LIBNPY_URL ${EXTERNAL_DOWNLOAD_SERVER}/llohse/libnpy/archive/refs/tags/v0.1.0.tar.gz)

  ExternalSource_Download(LIBNPY
    DIRNAME LIBNPY
    URL ${LIBNPY_URL})

  set(LibnpySource_DIR ${LIBNPY_SOURCE_DIR} PARENT_SCOPE)
  set(LibnpySource_FOUND TRUE PARENT_SCOPE)
endfunction(_LibnpySource_import)

_LibnpySource_import()
