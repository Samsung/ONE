function(_JsoncppSource_import)
  if(NOT DOWNLOAD_JSONCPP)
    set(JsoncppSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_JSONCPP)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(JSONCPP_URL ${EXTERNAL_DOWNLOAD_SERVER}/open-source-parsers/jsoncpp/archive/refs/tags/1.9.5.tar.gz)

  ExternalSource_Download(JSONCPP ${JSONCPP_URL})

  set(JsoncppSource_DIR ${JSONCPP_SOURCE_DIR} PARENT_SCOPE)
  set(JsoncppSource_FOUND TRUE PARENT_SCOPE)
endfunction(_JsoncppSource_import)

_JsoncppSource_import()
