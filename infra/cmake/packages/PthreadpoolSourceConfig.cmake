function(_PthreadpoolSource_import)
  if(NOT ${DOWNLOAD_PTHREADPOOL})
    set(PthreadpoolSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${DOWNLOAD_PTHREADPOOL})

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  # pthreadpool commit in xnnpack 8b283aa30a31
  envoption(PTHREADPOOL_URL ${EXTERNAL_DOWNLOAD_SERVER}/Maratyszcza/pthreadpool/archive/4fe0e1e183925bf8cfa6aae24237e724a96479b8.tar.gz)
  ExternalSource_Download(PTHREADPOOL
    DIRNAME PTHREADPOOL
    URL ${PTHREADPOOL_URL})

  set(PthreadpoolSource_DIR ${PTHREADPOOL_SOURCE_DIR} PARENT_SCOPE)
  set(PthreadpoolSource_FOUND TRUE PARENT_SCOPE)
endfunction(_PthreadpoolSource_import)

_PthreadpoolSource_import()
