function(_EigenSource_import)
  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # NOTE TensorFlow 1.12 downloads Eign from the following URL
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://bitbucket.org")
  set(EIGEN_URL ${EXTERNAL_DOWNLOAD_SERVER}/eigen/eigen/get/88fc23324517.tar.gz)
  ExternalSource_Get("eigen" ${DOWNLOAD_EIGEN} ${EIGEN_URL})

  set(EigenSource_DIR ${eigen_SOURCE_DIR} PARENT_SCOPE)
  set(EigenSource_FOUND ${eigen_SOURCE_GET} PARENT_SCOPE)
endfunction(_EigenSource_import)

_EigenSource_import()
