function(_CMSISSource_import)
  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(CMSIS_5_8_0_URL ${EXTERNAL_DOWNLOAD_SERVER}/ARM-software/CMSIS_5/archive/refs/tags/5.8.0.tar.gz)
  set(CMSIS_5_8_0_SHA256 fe6b697b8782e7fd6131034b7646a3b65c83018774abf7f9f94901a3bc7c82ad)

  ExternalSource_Download(CMSIS DIRNAME CMSIS-5.8.0 ${CMSIS_5_8_0_URL}
          CHECKSUM "SHA256=${CMSIS_5_8_0_SHA256}")

  set(CMSISSource_DIR ${CMSIS_SOURCE_DIR} PARENT_SCOPE)
  set(CMSISSource_FOUND TRUE PARENT_SCOPE)
endfunction(_CMSISSource_import)

_CMSISSource_import()
