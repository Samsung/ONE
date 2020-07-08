function(_TensorFlowSource_import)
  if(NOT DOWNLOAD_TENSORFLOW)
    set(TensorFlowSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_TENSORFLOW)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(TENSORFLOW_1_13_1_URL ${EXTERNAL_DOWNLOAD_SERVER}/tensorflow/tensorflow/archive/v1.13.1.tar.gz)

  ExternalSource_Download(TENSORFLOW DIRNAME TENSORFLOW-1.13.1 ${TENSORFLOW_1_13_1_URL})

  set(TensorFlowSource_DIR ${TENSORFLOW_SOURCE_DIR} PARENT_SCOPE)
  set(TensorFlowSource_FOUND TRUE PARENT_SCOPE)
endfunction(_TensorFlowSource_import)

_TensorFlowSource_import()
