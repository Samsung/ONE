function(_NNPACKSource_import)
    if(NOT DOWNLOAD_NNPACK)
        set(NNPACKSource_FOUND FALSE PARENT_SCOPE)
        message(WARNING "NNPACK not downloaded")
        return()
    endif(NOT DOWNLOAD_NNPACK)

    nnas_include(ExternalSourceTools)
    nnas_include(OptionTools)

    envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
    set(NNPACK_URL ${EXTERNAL_DOWNLOAD_SERVER}/Maratyszcza/NNPACK/archive/c039579abe21f5756e0f0e45e8e767adccc11852.tar.gz)
    ExternalSource_Get("NNPACK" ${DOWNLOAD_NNPACK} ${NNPACK_URL})

    set(NNPACK_SOURCE_DIR ${NNPACK_SOURCE_DIR} PARENT_SCOPE)
    set(NNPACK_INCLUDE_DIR ${NNPACK_SOURCE_DIR}/include PARENT_SCOPE)
    set(NNPACK_SOURCE_FOUND ${NNPACK_SOURCE_GET} PARENT_SCOPE)
endfunction(_NNPACKSource_import)

_NNPACKSource_import()
