function(_TensorFlowRuySource_import)
  if(NOT DOWNLOAD_RUY)
    set(TensorFlowRuySource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_RUY)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # Exact version used by TensorFlow v2.6.0.
  # See tensorflow/third_party/ruy/workspace.bzl
  envoption(TENSORFLOW_2_6_0_RUY_URL https://github.com/google/ruy/archive/e6c1b8dc8a8b00ee74e7268aac8b18d7260ab1ce.zip)

  ExternalSource_Download(RUY DIRNAME TENSORFLOW-2.6.0-RUY ${TENSORFLOW_2_6_0_RUY_URL})

  set(TensorFlowRuySource_DIR ${RUY_SOURCE_DIR} PARENT_SCOPE)
  set(TensorFlowRuySource_FOUND TRUE PARENT_SCOPE)
endfunction(_TensorFlowRuySource_import)

_TensorFlowRuySource_import()
