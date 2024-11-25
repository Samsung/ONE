function(_RobinMapSource_import)
  if(NOT DOWNLOAD_ROBINMAP)
    set(RobinMapSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_ROBINMAP)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(ROBINMAP_URL ${EXTERNAL_DOWNLOAD_SERVER}/Tessil/robin-map/archive/v1.3.0.tar.gz)

  ExternalSource_Download(ROBINMAP ${ROBINMAP_URL})

  set(RobinMapSource_DIR ${ROBINMAP_SOURCE_DIR} PARENT_SCOPE)
  set(RobinMapSource_FOUND TRUE PARENT_SCOPE)
endfunction(_RobinMapSource_import)

_RobinMapSource_import()
