function(_NanobindSource_import)
  if(NOT DOWNLOAD_NANOBIND)
    set(NanobindSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_NANOBIND)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(NANOBIND_URL ${EXTERNAL_DOWNLOAD_SERVER}/wjakob/nanobind/archive/v2.2.0.tar.gz)

  ExternalSource_Download(NANOBIND ${NANOBIND_URL})

  set(NanobindSource_DIR ${NANOBIND_SOURCE_DIR} PARENT_SCOPE)
  set(NanobindSource_FOUND TRUE PARENT_SCOPE)
endfunction(_NanobindSource_import)

_NanobindSource_import()
