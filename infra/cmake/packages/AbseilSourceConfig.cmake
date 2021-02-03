function(_AbseilSource_import)
  if(NOT DOWNLOAD_ABSEIL)
    set(AbseilSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_ABSEIL)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # NOTE TensorFlow 2.3 downloads abseil from the following URL
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(ABSEIL_URL ${EXTERNAL_DOWNLOAD_SERVER}/abseil/abseil-cpp/archive/df3ea785d8c30a9503321a3d35ee7d35808f190d.tar.gz)

  ExternalSource_Download(ABSEIL
    DIRNAME ABSEIL
    URL ${ABSEIL_URL}
    CHECKSUM MD5=4d9aa7e757adf48fef171c85f0d88552)

  set(AbseilSource_DIR ${ABSEIL_SOURCE_DIR} PARENT_SCOPE)
  set(AbseilSource_FOUND TRUE PARENT_SCOPE)
endfunction(_AbseilSource_import)

_AbseilSource_import()
