# find_package rejects version with extra string like "2.3.0-rc0"
#
# TODO Find a better way
function(_import)
  if(NOT DOWNLOAD_TENSORFLOW)
    set(TensorFlowSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_TENSORFLOW)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(TENSORFLOW_2_3_0_RC0_URL https://github.com/tensorflow/tensorflow/archive/v2.3.0-rc0.tar.gz)

  ExternalSource_Download(TENSORFLOW DIRNAME TENSORFLOW-2.3.0-RC0 ${TENSORFLOW_2_3_0_RC0_URL})

  set(TensorFlowSource_DIR ${TENSORFLOW_SOURCE_DIR} PARENT_SCOPE)
  set(TensorFlowSource_FOUND TRUE PARENT_SCOPE)
endfunction(_import)

_import()
