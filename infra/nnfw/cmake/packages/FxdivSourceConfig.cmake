function(_FxdivSource_import)
  if(NOT ${DOWNLOAD_FXDIV})
    set(FxdivSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${DOWNLOAD_FXDIV})

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  # fxdiv commit in tflite v2.16.1
  envoption(FXDIV_URL ${EXTERNAL_DOWNLOAD_SERVER}/Maratyszcza/FXdiv/archive/63058eff77e11aa15bf531df5dd34395ec3017c8.tar.gz)
  ExternalSource_Download(FXDIV
    DIRNAME FXDIV
    URL ${FXDIV_URL})

  set(FxdivSource_DIR ${FXDIV_SOURCE_DIR} PARENT_SCOPE)
  set(FxdivSource_FOUND TRUE PARENT_SCOPE)
endfunction(_FxdivSource_import)

_FxdivSource_import()
