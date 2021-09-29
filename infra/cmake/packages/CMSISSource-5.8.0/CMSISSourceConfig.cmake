function(_CMSISSource_import)
  if(NOT DOWNLOAD_CMSIS)
    set(CMSIS_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_CMSIS)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(CMSIS_5_8_0_URL https://github.com/ARM-software/CMSIS_5/archive/refs/tags/5.8.0.tar.gz)

  ExternalSource_Download(CMSIS DIRNAME CMSIS-5.8.0 ${CMSIS_5_8_0_URL})

  set(CMSISSource_DIR ${CMSIS_SOURCE_DIR} PARENT_SCOPE)
  set(CMSISSource_FOUND TRUE PARENT_SCOPE)
endfunction(_CMSISSource_import)

_CMSISSource_import()
