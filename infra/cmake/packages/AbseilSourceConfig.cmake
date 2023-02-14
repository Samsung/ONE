function(_AbseilSource_import)
  if(NOT DOWNLOAD_ABSEIL)
    set(AbseilSource_FOUND
        FALSE
        PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_ABSEIL)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # NOTE TensorFlow 2.9 downloads abseil 20211102.0
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(ABSEIL_URL ${EXTERNAL_DOWNLOAD_SERVER}/abseil/abseil-cpp/archive/20211102.0.tar.gz)
  ExternalSource_Download(
    ABSEIL
    DIRNAME
    ABSEIL
    URL
    ${ABSEIL_URL}
    CHECKSUM
    MD5=bdca561519192543378b7cade101ec43)

  set(AbseilSource_DIR
      ${ABSEIL_SOURCE_DIR}
      PARENT_SCOPE)
  set(AbseilSource_FOUND
      TRUE
      PARENT_SCOPE)
endfunction(_AbseilSource_import)

_AbseilSource_import()
