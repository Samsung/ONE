function(_cpuinfoSource_import)
    nnas_include(ExternalSourceTools)
    nnas_include(OptionTools)

    envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
    set(CPUINFO_URL ${EXTERNAL_DOWNLOAD_SERVER}/pytorch/cpuinfo/archive/d5e37adf1406cf899d7d9ec1d317c47506ccb970.tar.gz)

    ExternalSource_Get("CPUINFO" ${DOWNLOAD_NNPACK} ${CPUINFO_URL})

    set(CPUINFO_SOURCE_DIR ${CPUINFO_SOURCE_DIR} PARENT_SCOPE)
    set(CPUINFO_SOURCE_FOUND ${CPUINFO_SOURCE_GET} PARENT_SCOPE)
endfunction(_cpuinfoSource_import)

_cpuinfoSource_import()
