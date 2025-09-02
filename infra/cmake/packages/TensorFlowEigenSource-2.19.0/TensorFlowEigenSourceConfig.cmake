function(_TensorFlowEigenSource_import)
  if(NOT DOWNLOAD_EIGEN)
    set(TensorFlowEigenSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_EIGEN)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # Exact version used by TensorFlow v2.19.0.
  # See tensorflow/third_party/eigen3/workspace.bzl.
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://gitlab.com")
  envoption(TENSORFLOW_2_19_0_EIGEN_URL ${EXTERNAL_DOWNLOAD_SERVER}/libeigen/eigen/-/archive/33d0937c6bdf5ec999939fb17f2a553183d14a74/eigen-33d0937c6bdf5ec999939fb17f2a553183d14a74.tar.gz)

  ExternalSource_Download(EIGEN DIRNAME TENSORFLOW-2.19.0-EIGEN ${TENSORFLOW_2_19_0_EIGEN_URL})

  set(TensorFlowEigenSource_DIR ${EIGEN_SOURCE_DIR} PARENT_SCOPE)
  set(TensorFlowEigenSource_FOUND TRUE PARENT_SCOPE)
endfunction(_TensorFlowEigenSource_import)

_TensorFlowEigenSource_import()
