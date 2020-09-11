function(_CpuInfoSource_import)
  if(NOT ${DOWNLOAD_CPUINFO})
    set(CpuInfoSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${DOWNLOAD_CPUINFO})

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(CPUINFO_URL ${EXTERNAL_DOWNLOAD_SERVER}/pytorch/cpuinfo/archive/5cefcd6293e6881754c2c53f99e95b159d2d8aa5.zip)
  ExternalSource_Download(CPUINFO
    DIRNAME CPUINFO
    URL ${CPUINFO_URL})

  # Applying patch to cpuinfo (This patch comes from tflive v2.3)
  execute_process(
    COMMAND patch -p1 --forward --ignore-whitespace
    WORKING_DIRECTORY ${CPUINFO_SOURCE_DIR}
    INPUT_FILE "${CMAKE_CURRENT_LIST_DIR}/CpuInfo/cpuinfo.patch"
    OUTPUT_VARIABLE OUTPUT
    RESULT_VARIABLE RESULT
  )

  set(CpuInfoSource_DIR ${CPUINFO_SOURCE_DIR} PARENT_SCOPE)
  set(CpuInfoSource_FOUND TRUE PARENT_SCOPE)
endfunction(_CpuInfoSource_import)

_CpuInfoSource_import()
