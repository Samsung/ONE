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
  # Let's use the latest released version (2018-12 release)
  envoption(ABSEIL_URL https://github.com/abseil/abseil-cpp/archive/20181200.tar.gz)

  ExternalSource_Download(ABSEIL ${ABSEIL_URL})

  set(AbseilSource_DIR ${ABSEIL_SOURCE_DIR} PARENT_SCOPE)
  set(AbseilSource_FOUND TRUE PARENT_SCOPE)
endfunction(_AbseilSource_import)

_AbseilSource_import()
