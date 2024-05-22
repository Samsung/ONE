function(_TensorFlowGEMMLowpSource_import)
  if(NOT DOWNLOAD_GEMMLOWP)
    set(TensorFlowGEMMLowpSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_GEMMLOWP)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # Exact version used by TensorFlow v2.16.1.
  # See tensorflow/third_party/gemmlowp/workspace.bzl.
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(TENSORFLOW_2_16_1_GEMMLOWP_URL ${EXTERNAL_DOWNLOAD_SERVER}/google/gemmlowp/archive/16e8662c34917be0065110bfcd9cc27d30f52fdf.zip)

  ExternalSource_Download(GEMMLOWP DIRNAME TENSORFLOW-2.16.1-GEMMLOWP ${TENSORFLOW_2_16_1_GEMMLOWP_URL})

  set(TensorFlowGEMMLowpSource_DIR ${GEMMLOWP_SOURCE_DIR} PARENT_SCOPE)
  set(TensorFlowGEMMLowpSource_FOUND TRUE PARENT_SCOPE)
endfunction(_TensorFlowGEMMLowpSource_import)

_TensorFlowGEMMLowpSource_import()
