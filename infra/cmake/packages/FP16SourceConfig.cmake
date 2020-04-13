function(_FP16Source_import)
    nnas_include(ExternalSourceTools)
    nnas_include(OptionTools)

    envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
    set(FP16_URL ${EXTERNAL_DOWNLOAD_SERVER}/Maratyszcza/FP16/archive/febbb1c163726b5db24bed55cc9dc42529068997.tar.gz)

    ExternalSource_Get("FP16" ${DOWNLOAD_NNPACK} ${FP16_URL})

    set(FP16_SOURCE_DIR ${FP16_SOURCE_DIR} PARENT_SCOPE)
    set(FP16_SOURCE_FOUND ${FP16_SOURCE_GET} PARENT_SCOPE)
endfunction(_FP16Source_import)

_FP16Source_import()
