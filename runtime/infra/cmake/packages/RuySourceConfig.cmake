function(_RuySource_import)
  if(NOT DOWNLOAD_RUY)
    set(RuySource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_RUY)

  nnfw_include(ExternalSourceTools)
  nnfw_include(OptionTools)

  # Latest version: Sep.2025
  # Build with profiler option fail on exact version used by TensorFlow v2.19.1
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(RUY_URL ${EXTERNAL_DOWNLOAD_SERVER}/google/ruy/archive/9940fbf1e0c0863907e77e0600b99bb3e2bc2b9f.zip)
  #envoption(RUY_URL ${EXTERNAL_DOWNLOAD_SERVER}/google/ruy/archive/3286a34cc8de6149ac6844107dfdffac91531e72.zip)

  ExternalSource_Download(RUY DIRNAME RUY ${RUY_URL})

  set(RuySource_DIR ${RUY_SOURCE_DIR} PARENT_SCOPE)
  set(RuySource_FOUND TRUE PARENT_SCOPE)
endfunction(_RuySource_import)

_RuySource_import()
