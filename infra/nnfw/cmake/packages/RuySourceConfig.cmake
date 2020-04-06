function(_RuySource_import)
  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # NOTE TensorFlow master (2020-04-06) downloads Ruy from the following URL
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  set(RUY_URL ${EXTERNAL_DOWNLOAD_SERVER}/google/ruy/archive/91d62808498cea7ccb48aa59181e218b4ad05701.zip)
  ExternalSource_Get("ruy" ${DOWNLOAD_RUY} ${RUY_URL})

  set(RuySource_DIR ${ruy_SOURCE_DIR} PARENT_SCOPE)
  set(RuySource_FOUND ${ruy_SOURCE_GET} PARENT_SCOPE)
endfunction(_RuySource_import)

_RuySource_import()
