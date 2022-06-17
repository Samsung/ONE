function(_TensorFlowGEMMLowpSource_import)
  if(NOT DOWNLOAD_GEMMLOWP)
    set(TensorFlowGEMMLowpSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_GEMMLOWP)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # Exact version used by TensorFlow v2.1.0.
  # See tensorflow/tensorflow/workspace.bzl.
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(TENSORFLOW_2_1_0_GEMMLOWP_URL ${EXTERNAL_DOWNLOAD_SERVER}/google/gemmlowp/archive/12fed0cd7cfcd9e169bf1925bc3a7a58725fdcc3.zip)

  ExternalSource_Download(GEMMLOWP DIRNAME TENSORFLOW-2.1.0-GEMMLOWP ${TENSORFLOW_2_1_0_GEMMLOWP_URL})

  set(TensorFlowGEMMLowpSource_DIR ${GEMMLOWP_SOURCE_DIR} PARENT_SCOPE)
  set(TensorFlowGEMMLowpSource_FOUND TRUE PARENT_SCOPE)
endfunction(_TensorFlowGEMMLowpSource_import)

_TensorFlowGEMMLowpSource_import()
