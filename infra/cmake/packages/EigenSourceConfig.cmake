function(_EigenSource_import)
  if(NOT DOWNLOAD_EIGEN)
    set(EigenSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_EIGEN)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # NOTE TensorFlow 1.13.1 uses https://bitbucket.org/eigen/eigen/get/9f48e814419e.tar.gz
  #      The following URL fix bug above URL
  envoption(EIGEN_1_13_1_URL https://bitbucket.org/eigen/eigen/get/88fc23324517.tar.gz)

  ExternalSource_Download(EIGEN ${EIGEN_1_13_1_URL})

  set(EigenSource_DIR ${EIGEN_SOURCE_DIR} PARENT_SCOPE)
  set(EigenSource_FOUND TRUE PARENT_SCOPE)
endfunction(_EigenSource_import)

_EigenSource_import()
