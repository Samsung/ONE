function(_TensorFlowGEMMLowpSource_import)
  if(NOT DOWNLOAD_GEMMLOWP)
    set(TensorFlowGEMMLowpSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_GEMMLOWP)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # Exact version used by TensorFlow v2.15.0
  # See tensorflow/third_party/gemmlowp/workspace.bzl.
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(TENSORFLOW_2_15_0_GEMMLOWP_URL ${EXTERNAL_DOWNLOAD_SERVER}/google/gemmlowp/archive/e844ffd17118c1e17d94e1ba4354c075a4577b88.zip)

  ExternalSource_Download(GEMMLOWP DIRNAME TENSORFLOW-2.15.0-GEMMLOWP ${TENSORFLOW_2_15_0_GEMMLOWP_URL})

  set(TensorFlowGEMMLowpSource_DIR ${GEMMLOWP_SOURCE_DIR} PARENT_SCOPE)
  set(TensorFlowGEMMLowpSource_FOUND TRUE PARENT_SCOPE)
endfunction(_TensorFlowGEMMLowpSource_import)

_TensorFlowGEMMLowpSource_import()
