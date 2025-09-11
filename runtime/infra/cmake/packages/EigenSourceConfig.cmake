function(_EigenSource_import)
  if(NOT DOWNLOAD_EIGEN)
    set(EigenSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_EIGEN)

  nnfw_include(ExternalSourceTools)
  nnfw_include(OptionTools)

  # Exact version used by TensorFlow v2.19.1.
  # See tensorflow/third_party/eigen3/workspace.bzl.
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://gitlab.com")
  envoption(EIGEN_URL ${EXTERNAL_DOWNLOAD_SERVER}/libeigen/eigen/-/archive/33d0937c6bdf5ec999939fb17f2a553183d14a74/eigen-33d0937c6bdf5ec999939fb17f2a553183d14a74.tar.gz)

  ExternalSource_Download(EIGEN DIRNAME EIGEN ${EIGEN_URL})

  set(EigenSource_DIR ${EIGEN_SOURCE_DIR} PARENT_SCOPE)
  set(EigenSource_FOUND TRUE PARENT_SCOPE)
endfunction(_EigenSource_import)

_EigenSource_import()
