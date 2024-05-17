function(_Fp16Source_import)
  if(NOT ${DOWNLOAD_FP16})
    set(Fp16Source_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${DOWNLOAD_FP16})

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  # fp16 commit in xnnpack (tflite v2.16.1)
  envoption(FP16_URL ${EXTERNAL_DOWNLOAD_SERVER}/Maratyszcza/FP16/archive/0a92994d729ff76a58f692d3028ca1b64b145d91.tar.gz)
  ExternalSource_Download(FP16
    DIRNAME FP16
    URL ${FP16_URL})

  set(Fp16Source_DIR ${FP16_SOURCE_DIR} PARENT_SCOPE)
  set(Fp16Source_FOUND TRUE PARENT_SCOPE)
endfunction(_Fp16Source_import)

_Fp16Source_import()
