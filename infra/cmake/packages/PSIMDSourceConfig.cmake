function(_PSIMDSource_import)
    nnas_include(ExternalSourceTools)
    nnas_include(OptionTools)

    envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
    set(PSIMD_URL ${EXTERNAL_DOWNLOAD_SERVER}/Maratyszcza/psimd/archive/90a938f30ba414ada2f4b00674ee9631d7d85e19.tar.gz)

    ExternalSource_Get("PSIMD" ${DOWNLOAD_NNPACK} ${PSIMD_URL})

    set(PSIMD_SOURCE_DIR ${PSIMD_SOURCE_DIR} PARENT_SCOPE)
    set(PSIMD_SOURCE_FOUND ${PSIMD_SOURCE_GET} PARENT_SCOPE)
endfunction(_PSIMDSource_import)

_PSIMDSource_import()
