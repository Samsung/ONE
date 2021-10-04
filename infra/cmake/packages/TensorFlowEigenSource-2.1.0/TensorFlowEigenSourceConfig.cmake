function(_TensorFlowEigenSource_import)
  if(NOT DOWNLOAD_EIGEN)
    set(TensorFlowEigenSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_EIGEN)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # Exact version used by TensorFlow v2.1.0.
  # See tensorflow/tensorflow/workspace.bzl.
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://eigen.googlesource.com")
  envoption(TENSORFLOW_2_1_0_EIGEN_URL ${EXTERNAL_DOWNLOAD_SERVER}/mirror/+archive/4e696901f873a2347f76d931cf2f701e31e15d05.tar.gz)

  ExternalSource_Download(EIGEN DIRNAME TENSORFLOW-2.1.0-EIGEN ${TENSORFLOW_2_1_0_EIGEN_URL})

  set(TensorFlowEigenSource_DIR ${EIGEN_SOURCE_DIR} PARENT_SCOPE)
  set(TensorFlowEigenSource_FOUND TRUE PARENT_SCOPE)
endfunction(_TensorFlowEigenSource_import)

_TensorFlowEigenSource_import()
