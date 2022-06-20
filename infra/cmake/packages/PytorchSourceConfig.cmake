function(_PytorchSource_import)
  if(NOT DOWNLOAD_PYTORCH)
    set(PytorchSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_PYTORCH)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(PYTORCH_URL ${EXTERNAL_DOWNLOAD_SERVER}/pytorch/pytorch/archive/v0.4.1.tar.gz)

  ExternalSource_Download(PYTORCH ${PYTORCH_URL})

  set(PytorchSource_DIR ${PYTORCH_SOURCE_DIR} PARENT_SCOPE)
  set(PytorchSource_FOUND ${DOWNLOAD_PYTORCH} PARENT_SCOPE)
endfunction(_PytorchSource_import)

_PytorchSource_import()
