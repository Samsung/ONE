function(_TensorFlowSource_import)
  if(NOT DOWNLOAD_TENSORFLOW)
    set(TensorFlowSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_TENSORFLOW)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  set(TENSORFLOW_URL ${EXTERNAL_DOWNLOAD_SERVER}/tensorflow/tensorflow/archive/v1.13.1.tar.gz)
  ExternalSource_Download("tensorflow" ${TENSORFLOW_URL})

  set(TensorFlowSource_DIR ${tensorflow_SOURCE_DIR} PARENT_SCOPE)
  set(TensorFlowSource_FOUND ${tensorflow_SOURCE_GET} PARENT_SCOPE)
endfunction(_TensorFlowSource_import)

_TensorFlowSource_import()
