function(_FxdivSource_import)
  if(NOT ${DOWNLOAD_FXDIV})
    set(FxdivSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${DOWNLOAD_FXDIV})

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  # fxdiv commit in xnnpack 8b283aa30a31
  envoption(FXDIV_URL ${EXTERNAL_DOWNLOAD_SERVER}/Maratyszcza/FXdiv/archive/f8c5354679ec2597792bc70a9e06eff50c508b9a.tar.gz)
  ExternalSource_Download(FXDIV
    DIRNAME FXDIV
    URL ${FXDIV_URL})

  set(FxdivSource_DIR ${FXDIV_SOURCE_DIR} PARENT_SCOPE)
  set(FxdivSource_FOUND TRUE PARENT_SCOPE)
endfunction(_FxdivSource_import)

_FxdivSource_import()
