function(_CMSIS_NN_import)
    nnas_include(ExternalSourceTools)
    nnas_include(OptionTools)

    envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
    envoption(CMSIS_NN_6_0_0_URL ${EXTERNAL_DOWNLOAD_SERVER}/ARM-software/CMSIS-NN/archive/refs/tags/v6.0.0.tar.gz)

    ExternalSource_Download(CMSIS_NN DIRNAME CMSIS-NN-6.0.0 ${CMSIS_NN_6_0_0_URL})

    set(CMSIS_NNSource_DIR ${CMSIS_NN_SOURCE_DIR} PARENT_SCOPE)
    set(CMSIS_NNSource_FOUND TRUE PARENT_SCOPE)
endfunction(_CMSIS_NN_import)

_CMSIS_NN_import()
