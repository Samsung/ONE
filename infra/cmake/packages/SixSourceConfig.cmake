function(_SIXSource_import)
    nnas_include(ExternalSourceTools)
    nnas_include(OptionTools)

    envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
    set(six_URL ${EXTERNAL_DOWNLOAD_SERVER}/benjaminp/six/archive/1.11.0.tar.gz)

    ExternalSource_Get("PYTHON_SIX" ${DOWNLOAD_NNPACK} ${six_URL})

    set(PYTHON_SIX_SOURCE_DIR ${PYTHON_SIX_SOURCE_DIR} PARENT_SCOPE)
    set(PYTHON_SIX_SOURCE_FOUND ${PYTHON_SIX_SOURCE_GET} PARENT_SCOPE)
endfunction(_SIXSource_import)

_SIXSource_import()
