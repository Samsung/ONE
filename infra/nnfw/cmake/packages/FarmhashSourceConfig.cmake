function(_FarmhashSource_import)
  if(NOT ${DOWNLOAD_FARMHASH})
    set(FarmhashSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${DOWNLOAD_FARMHASH})

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # NOTE TensorFlow 1.12 downloads farmhash from the following URL
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  set(FARMHASH_URL ${EXTERNAL_DOWNLOAD_SERVER}/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz)
  ExternalSource_Download("farmhash" ${FARMHASH_URL})

  set(FarmhashSource_DIR ${farmhash_SOURCE_DIR} PARENT_SCOPE)
  set(FarmhashSource_FOUND TRUE PARENT_SCOPE)
endfunction(_FarmhashSource_import)

_FarmhashSource_import()
