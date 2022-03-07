function(_GTestSource_import)
  if(NOT DOWNLOAD_GTEST)
    set(GTestSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_GTEST)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(GTEST_URL https://github.com/google/googletest/archive/release-1.11.0.tar.gz)

  ExternalSource_Download(GTEST ${GTEST_URL})

  set(GTestSource_DIR ${GTEST_SOURCE_DIR} PARENT_SCOPE)
  set(GTestSource_FOUND TRUE PARENT_SCOPE)
endfunction(_GTestSource_import)

_GTestSource_import()
