function(_CpuInfoSource_import)
  if(NOT ${DOWNLOAD_CPUINFO})
    set(CpuInfoSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${DOWNLOAD_CPUINFO})

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  # CPUINFO commit from tflite v2.16.1
  envoption(CPUINFO_URL ${EXTERNAL_DOWNLOAD_SERVER}/pytorch/cpuinfo/archive/ef634603954d88d2643d5809011288b890ac126e.tar.gz)
  ExternalSource_Download(CPUINFO
    DIRNAME CPUINFO
    URL ${CPUINFO_URL})

  set(CpuInfoSource_DIR ${CPUINFO_SOURCE_DIR} PARENT_SCOPE)
  set(CpuInfoSource_FOUND TRUE PARENT_SCOPE)
endfunction(_CpuInfoSource_import)

_CpuInfoSource_import()
