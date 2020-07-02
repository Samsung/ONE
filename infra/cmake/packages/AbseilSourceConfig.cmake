function(_AbseilSource_import)
  if(NOT DOWNLOAD_ABSEIL)
    set(AbseilSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_ABSEIL)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # NOTE TensorFlow 1.12 downloads abseil from the following URL
  # - https://github.com/abseil/abseil-cpp/archive/48cd2c3f351ff188bc85684b84a91b6e6d17d896.tar.gz
  #
  # The last change of "48cd2c3f351" was commited on 2018.09.27
  #
  # Let's use the latest released version (2020-02 release patch 2)
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(ABSEIL_URL ${EXTERNAL_DOWNLOAD_SERVER}/abseil/abseil-cpp/archive/20200225.2.tar.gz)

  ExternalSource_Download(ABSEIL
    DIRNAME ABSEIL
    URL ${ABSEIL_URL}
    CHECKSUM MD5=73f2b6e72f1599a9139170c29482ddc4)

  set(AbseilSource_DIR ${ABSEIL_SOURCE_DIR} PARENT_SCOPE)
  set(AbseilSource_FOUND TRUE PARENT_SCOPE)
endfunction(_AbseilSource_import)

_AbseilSource_import()
