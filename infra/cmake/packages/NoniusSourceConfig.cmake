function(_NoniusSource_import)
  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  set(NONIUS_URL ${EXTERNAL_DOWNLOAD_SERVER}/libnonius/nonius/archive/v1.2.0-beta.1.tar.gz)
  ExternalSource_Get("NONIUS" ${DOWNLOAD_NONIUS} ${NONIUS_URL})

  set(NoniusSource_DIR ${NONIUS_SOURCE_DIR} PARENT_SCOPE)
  set(NoniusSource_FOUND ${NONIUS_SOURCE_GET} PARENT_SCOPE)
endfunction(_NoniusSource_import)

_NoniusSource_import()
