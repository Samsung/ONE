function(_TensorFlowRuySource_import)
  if(NOT DOWNLOAD_RUY)
    set(TensorFlowRuySource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_RUY)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # Exact version used by TensorFlow v2.12.1.
  # See tensorflow/third_party/ruy/workspace.bzl
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(TENSORFLOW_2_12_1_RUY_URL ${EXTERNAL_DOWNLOAD_SERVER}/google/ruy/archive/3286a34cc8de6149ac6844107dfdffac91531e72.zip)

  ExternalSource_Download(RUY DIRNAME TENSORFLOW-2.12.1-RUY ${TENSORFLOW_2_12_1_RUY_URL})

  set(TensorFlowRuySource_DIR ${RUY_SOURCE_DIR} PARENT_SCOPE)
  set(TensorFlowRuySource_FOUND TRUE PARENT_SCOPE)
endfunction(_TensorFlowRuySource_import)

_TensorFlowRuySource_import()
