function(_OouraFFTSource_import)
  if(NOT DOWNLOAD_OOURAFFT)
    set(OouraFFTSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_OOURAFFT)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # NOTE TensorFlow 2.3 downloads OOURAFFT from the following URL
  envoption(OOURAFFT_URL https://github.com/petewarden/OouraFFT/archive/v1.0.tar.gz)

  ExternalSource_Download(OOURAFFT ${OOURAFFT_URL})

  set(OouraFFTSource_DIR ${OOURAFFT_SOURCE_DIR} PARENT_SCOPE)
  set(OouraFFTSource_FOUND TRUE PARENT_SCOPE)
endfunction(_OouraFFTSource_import)

_OouraFFTSource_import()
