function(_CpuInfoSource_import)
  if(NOT ${DOWNLOAD_CPUINFO})
    set(CpuInfoSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${DOWNLOAD_CPUINFO})

  nnfw_include(ExternalSourceTools)
  nnfw_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  # CPUINFO commit for RISC-V bug fix
  envoption(CPUINFO_URL ${EXTERNAL_DOWNLOAD_SERVER}/pytorch/cpuinfo/archive/6543fec09b2f04ac4a666882998b534afc9c1349.tar.gz)
  ExternalSource_Download(CPUINFO
    DIRNAME CPUINFO
    URL ${CPUINFO_URL})

  set(CpuInfoSource_DIR ${CPUINFO_SOURCE_DIR} PARENT_SCOPE)
  set(CpuInfoSource_FOUND TRUE PARENT_SCOPE)
endfunction(_CpuInfoSource_import)

_CpuInfoSource_import()
