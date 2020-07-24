function(_ARMComputeSource_import)
  if(NOT ${DOWNLOAD_ARMCOMPUTE})
    set(ARMComputeSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${DOWNLOAD_ARMCOMPUTE})

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  set(ARMCOMPUTE_URL ${EXTERNAL_DOWNLOAD_SERVER}/ARM-software/ComputeLibrary/archive/v19.11.1.tar.gz)
  ExternalSource_Download(ARMCOMPUTE ${ARMCOMPUTE_URL})

  set(ARMComputeSource_DIR ${ARMCOMPUTE_SOURCE_DIR} PARENT_SCOPE)
  set(ARMComputeSource_FOUND ${ARMCOMPUTE_SOURCE_GET} PARENT_SCOPE)
endfunction(_ARMComputeSource_import)

_ARMComputeSource_import()
