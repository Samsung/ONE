function(_ONNXSource_import)
  if(NOT DOWNLOAD_ONNX)
    set(ONNXSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_ONNX)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(ONNX_1_6_0_URL ${EXTERNAL_DOWNLOAD_SERVER}/onnx/onnx/archive/v1.6.0.zip)

  ExternalSource_Download(ONNX DIRNAME ONNX-1.6.0
                               CHECKSUM MD5=cbdc547a527f1b59c7f066c8d258b966
                               URL ${ONNX_1_6_0_URL})

  set(ONNXSource_DIR ${ONNX_SOURCE_DIR} PARENT_SCOPE)
  set(ONNXSource_FOUND TRUE PARENT_SCOPE)
endfunction(_ONNXSource_import)

_ONNXSource_import()
