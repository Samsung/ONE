function(_RuySource_import)
  if(NOT ${DOWNLOAD_RUY})
    set(RuySource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${DOWNLOAD_RUY})

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # NOTE Downloads ruy source used by tensorflow v2.8.0
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(RUY_URL ${EXTERNAL_DOWNLOAD_SERVER}/google/ruy/archive/e6c1b8dc8a8b00ee74e7268aac8b18d7260ab1ce.tar.gz)
  ExternalSource_Download(RUY
    DIRNAME RUY
    URL ${RUY_URL})

  set(RuySource_DIR ${RUY_SOURCE_DIR} PARENT_SCOPE)
  set(RuySource_FOUND TRUE PARENT_SCOPE)
endfunction(_RuySource_import)

_RuySource_import()
