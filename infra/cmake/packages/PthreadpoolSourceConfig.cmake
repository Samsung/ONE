function(_pthreadpoolSource_import)
    nnas_include(ExternalSourceTools)
    nnas_include(OptionTools)

    envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
    set(PTHREADPOOL_URL ${EXTERNAL_DOWNLOAD_SERVER}/Maratyszcza/pthreadpool/archive/6673a4c71fe35e077c6843a74017d9c25610c537.tar.gz)

    ExternalSource_Get("PTHREADPOOL" ${DOWNLOAD_NNPACK} ${PTHREADPOOL_URL})

    set(PTHREADPOOL_SOURCE_DIR ${PTHREADPOOL_SOURCE_DIR} PARENT_SCOPE)
    set(PTHREADPOOL_SOURCE_FOUND ${PTHREADPOOL_SOURCE_GET} PARENT_SCOPE)
endfunction(_pthreadpoolSource_import)

_pthreadpoolSource_import()
