function(_ONNXSource_import)
  if(NOT DOWNLOAD_ONNX)
    set(ONNXSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_ONNX)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(ONNX_1_4_1_URL ${EXTERNAL_DOWNLOAD_SERVER}/onnx/onnx/archive/v1.4.1.zip)

  ExternalSource_Download(ONNX DIRNAME ONNX-1.4.1
                               CHECKSUM MD5=604b43a22fbc758f32ae9f3a4fb9d397
                               URL ${ONNX_1_4_1_URL})

  set(ONNXSource_DIR ${ONNX_SOURCE_DIR} PARENT_SCOPE)
  set(ONNXSource_FOUND TRUE PARENT_SCOPE)
endfunction(_ONNXSource_import)

_ONNXSource_import()
