function(_CpuInfoSource_import)
  if(NOT ${DOWNLOAD_CPUINFO})
    set(CpuInfoSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${DOWNLOAD_CPUINFO})

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  # CPUINFO commit including patch from tflite v2.3
  envoption(CPUINFO_URL ${EXTERNAL_DOWNLOAD_SERVER}/pytorch/cpuinfo/archive/63b254577ed77a8004a9be6ac707f3dccc4e1fd9.tar.gz)
  ExternalSource_Download(CPUINFO
    DIRNAME CPUINFO
    URL ${CPUINFO_URL})

  set(CpuInfoSource_DIR ${CPUINFO_SOURCE_DIR} PARENT_SCOPE)
  set(CpuInfoSource_FOUND TRUE PARENT_SCOPE)
endfunction(_CpuInfoSource_import)

_CpuInfoSource_import()
