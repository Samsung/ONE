function(_FXdivSource_import)
    nnas_include(ExternalSourceTools)
    nnas_include(OptionTools)

    envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
    set(FXDIV_URL ${EXTERNAL_DOWNLOAD_SERVER}/Maratyszcza/FXdiv/archive/f8c5354679ec2597792bc70a9e06eff50c508b9a.tar.gz)

    ExternalSource_Get("FXDIV" ${DOWNLOAD_NNPACK} ${FXDIV_URL})

    set(FXDIV_SOURCE_DIR ${FXDIV_SOURCE_DIR} PARENT_SCOPE)
    set(FXDIV_SOURCE_FOUND ${FXDIV_SOURCE_GET} PARENT_SCOPE)
endfunction(_FXdivSource_import)

_FXdivSource_import()
