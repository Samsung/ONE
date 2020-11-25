function(_XnnpackSource_import)
  if(NOT ${DOWNLOAD_XNNPACK})
    set(XnnpackSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${DOWNLOAD_XNNPACK})

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  # xnnpack commit in tflite v2.3
  envoption(XNNPACK_URL ${EXTERNAL_DOWNLOAD_SERVER}/google/XNNPACK/archive/8b283aa30a3186c6e640aed520543e9c067132d.tar.gz)
  ExternalSource_Download(XNNPACK
    DIRNAME XNNPACK
    URL ${XNNPACK_URL})

  set(XnnpackSource_DIR ${XNNPACK_SOURCE_DIR} PARENT_SCOPE)
  set(XnnpackSource_FOUND TRUE PARENT_SCOPE)
endfunction(_XnnpackSource_import)

_XnnpackSource_import()
