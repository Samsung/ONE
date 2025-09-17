function(_OouraFFTSource_import)
  if(NOT DOWNLOAD_OOURAFFT)
    set(OouraFFTSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_OOURAFFT)

  nnfw_include(ExternalSourceTools)
  nnfw_include(OptionTools)

  # NOTE TensorFlow 2.18.1 downloads OOURAFFT from the following URL
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(OOURAFFT_URL ${EXTERNAL_DOWNLOAD_SERVER}/petewarden/OouraFFT/archive/v1.0.tar.gz)

  ExternalSource_Download(OOURAFFT ${OOURAFFT_URL})

  set(OouraFFTSource_DIR ${OOURAFFT_SOURCE_DIR} PARENT_SCOPE)
  set(OouraFFTSource_FOUND TRUE PARENT_SCOPE)
endfunction(_OouraFFTSource_import)

_OouraFFTSource_import()
