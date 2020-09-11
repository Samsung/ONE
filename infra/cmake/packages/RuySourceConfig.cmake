function(_RuySource_import)
  if(NOT ${DOWNLOAD_RUY})
    set(RuySource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${DOWNLOAD_RUY})

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # NOTE Downloads source from latest ruy library (2020-04-10)
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(RUY_URL ${EXTERNAL_DOWNLOAD_SERVER}/google/ruy/archive/2e2658f964638ab7aa562d4b48b76007d44e38f0.tar.gz)
  ExternalSource_Download(RUY
    DIRNAME RUY
    URL ${RUY_URL})

  set(RuySource_DIR ${RUY_SOURCE_DIR} PARENT_SCOPE)
  set(RuySource_FOUND TRUE PARENT_SCOPE)
endfunction(_RuySource_import)

_RuySource_import()
