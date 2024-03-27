function(_FarmhashSource_import)
  if(NOT DOWNLOAD_FARMHASH)
    set(FarmhashSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_FARMHASH)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # NOTE TensorFlow 2.12.1 downloads farmhash from the following URL
  #      TensorFlow 2.15.0 downloads farmhash from the following URL
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(FARMHASH_URL ${EXTERNAL_DOWNLOAD_SERVER}/google/farmhash/archive/0d859a811870d10f53a594927d0d0b97573ad06d.tar.gz)

  ExternalSource_Download(FARMHASH ${FARMHASH_URL})

  set(FarmhashSource_DIR ${FARMHASH_SOURCE_DIR} PARENT_SCOPE)
  set(FarmhashSource_FOUND TRUE PARENT_SCOPE)
endfunction(_FarmhashSource_import)

_FarmhashSource_import()
