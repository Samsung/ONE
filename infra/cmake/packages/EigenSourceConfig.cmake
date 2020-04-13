function(_EigenSource_import)
  if(NOT DOWNLOAD_EIGEN)
    set(EigenSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_EIGEN)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # NOTE The following URL comes from TensorFlow 1.7
  envoption(EIGEN_URL https://bitbucket.org/eigen/eigen/get/2355b229ea4c.tar.gz)

  ExternalSource_Download(EIGEN ${EIGEN_URL})

  set(EigenSource_DIR ${EIGEN_SOURCE_DIR} PARENT_SCOPE)
  set(EigenSource_FOUND TRUE PARENT_SCOPE)
endfunction(_EigenSource_import)

_EigenSource_import()
