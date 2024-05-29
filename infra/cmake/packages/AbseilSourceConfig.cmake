function(_AbseilSource_import)
  if(NOT DOWNLOAD_ABSEIL)
    set(AbseilSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_ABSEIL)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # NOTE TensorFlow 2.16.1 downloads abseil from the following URL
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(ABSEIL_URL ${EXTERNAL_DOWNLOAD_SERVER}/abseil/abseil-cpp/archive/fb3621f4f897824c0dbe0615fa94543df6192f30.tar.gz)
  ExternalSource_Download(ABSEIL
    DIRNAME ABSEIL
    URL ${ABSEIL_URL})

  set(AbseilSource_DIR ${ABSEIL_SOURCE_DIR} PARENT_SCOPE)
  set(AbseilSource_FOUND TRUE PARENT_SCOPE)
endfunction(_AbseilSource_import)

_AbseilSource_import()
