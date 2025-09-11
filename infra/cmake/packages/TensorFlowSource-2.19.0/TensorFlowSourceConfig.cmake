function(_TensorFlowSource_import)
  if(NOT DOWNLOAD_TENSORFLOW)
    set(TensorFlowSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_TENSORFLOW)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(TENSORFLOW_2_19_0_URL ${EXTERNAL_DOWNLOAD_SERVER}/tensorflow/tensorflow/archive/v2.19.0.tar.gz)

  # Apply patch to build luci-compute without Ruy - comment out header include which is not using now
  ExternalSource_Download(TENSORFLOW
    DIRNAME TENSORFLOW-2.19.0
    PATCH ${CMAKE_CURRENT_LIST_DIR}/TensorFlowSource.patch
    ${TENSORFLOW_2_19_0_URL}
  )

  set(TensorFlowSource_DIR ${TENSORFLOW_SOURCE_DIR} PARENT_SCOPE)
  set(TensorFlowSource_FOUND TRUE PARENT_SCOPE)
endfunction(_TensorFlowSource_import)

_TensorFlowSource_import()
