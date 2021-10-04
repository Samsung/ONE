function(_TensorFlowEigenSource_import)
  if(NOT DOWNLOAD_EIGEN)
    set(TensorFlowEigenSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_EIGEN)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # Exact version used by TensorFlow v2.3.0.
  # See tensorflow/tensorflow/workspace.bzl.
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://eigen.googlesource.com")
  envoption(TENSORFLOW_2_3_0_EIGEN_URL ${EXTERNAL_DOWNLOAD_SERVER}/mirror/+archive/386d809bde475c65b7940f290efe80e6a05878c4.tar.gz)

  ExternalSource_Download(EIGEN DIRNAME TENSORFLOW-2.3.0-EIGEN ${TENSORFLOW_2_3_0_EIGEN_URL})

  set(TensorFlowEigenSource_DIR ${EIGEN_SOURCE_DIR} PARENT_SCOPE)
  set(TensorFlowEigenSource_FOUND TRUE PARENT_SCOPE)
endfunction(_TensorFlowEigenSource_import)

_TensorFlowEigenSource_import()
