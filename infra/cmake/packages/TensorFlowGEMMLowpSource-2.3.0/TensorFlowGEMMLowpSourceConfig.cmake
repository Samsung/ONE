function(_TensorFlowGEMMLowpSource_import)
  if(NOT DOWNLOAD_GEMMLOWP)
    set(TensorFlowGEMMLowpSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_GEMMLOWP)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # Exact version used by TensorFlow v2.3.0.
  # See tensorflow/tensorflow/workspace.bzl.
  envoption(TENSORFLOW_2_3_0_GEMMLOWP_URL https://github.com/google/gemmlowp/archive/fda83bdc38b118cc6b56753bd540caa49e570745.zip)

  ExternalSource_Download(GEMMLOWP DIRNAME TENSORFLOW-2.3.0-GEMMLOWP ${TENSORFLOW_2_3_0_GEMMLOWP_URL})

  set(TensorFlowGEMMLowpSource_DIR ${GEMMLOWP_SOURCE_DIR} PARENT_SCOPE)
  set(TensorFlowGEMMLowpSource_FOUND TRUE PARENT_SCOPE)
endfunction(_TensorFlowGEMMLowpSource_import)

_TensorFlowGEMMLowpSource_import()
