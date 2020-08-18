function(_TensorFlowRuySource_import)
  if(NOT DOWNLOAD_RUY)
    set(TensorFlowRuySource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_RUY)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # Exact version used by TensorFlow v2.3.0.
  # See tensorflow/third_party/ruy/workspace.bzl
  envoption(TENSORFLOW_2_3_0_RUY_URL https://github.com/google/ruy/archive/34ea9f4993955fa1ff4eb58e504421806b7f2e8f.zip)

  ExternalSource_Download(RUY DIRNAME TENSORFLOW-2.3.0-RUY ${TENSORFLOW_2_3_0_RUY_URL})

  set(TensorFlowRuySource_DIR ${RUY_SOURCE_DIR} PARENT_SCOPE)
  set(TensorFlowRuySource_FOUND TRUE PARENT_SCOPE)
endfunction(_TensorFlowRuySource_import)

_TensorFlowRuySource_import()
