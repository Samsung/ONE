function(_RuySource_import)
  if(NOT DOWNLOAD_RUY)
    set(RuySource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_RUY)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # Exact version used by TensorFlow v2.16.1.
  # See tensorflow/third_party/ruy/workspace.bzl
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(RUY_URL ${EXTERNAL_DOWNLOAD_SERVER}/google/ruy/archive/3286a34cc8de6149ac6844107dfdffac91531e72.zip)

  ExternalSource_Download(RUY DIRNAME TENSORFLOW-2.16.1-RUY ${RUY_URL})

  set(RuySource_DIR ${RUY_SOURCE_DIR} PARENT_SCOPE)
  set(RuySource_FOUND TRUE PARENT_SCOPE)
endfunction(_RuySource_import)

_RuySource_import()
