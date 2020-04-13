function(_enum34Source_import)
    nnas_include(ExternalSourceTools)
    nnas_include(OptionTools)

    envoption(EXTERNAL_DOWNLOAD_SERVER "https://bitbucket.org")
    set(ENUM34_URL ${EXTERNAL_DOWNLOAD_SERVER}/stoneleaf/enum34/get/1.1.6.tar.gz)

    ExternalSource_Get("PYTHON_ENUM" ${DOWNLOAD_NNPACK} ${ENUM34_URL})

    set(PYTHON_ENUM_SOURCE_DIR ${PYTHON_ENUM_SOURCE_DIR} PARENT_SCOPE)
    set(PYTHON_ENUM_SOURCE_FOUND ${PYTHON_ENUM_SOURCE_GET} PARENT_SCOPE)
endfunction(_enum34Source_import)

_enum34Source_import()
