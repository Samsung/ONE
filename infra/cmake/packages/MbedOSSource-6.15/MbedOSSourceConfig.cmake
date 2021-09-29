function(_MbedOSSource_import)
  if(NOT DOWNLOAD_MBEDOS)
    set(MbedOSSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_MBEDOS)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(MBEDOS_6_15_URL https://github.com/ARMmbed/mbed-os/archive/refs/tags/mbed-os-6.15.0.tar.gz)

  ExternalSource_Download(MBEDOS DIRNAME MBEDOS-6.15 ${MBEDOS_6_15_URL})

  set(MbedOSSource_DIR ${MBEDOS_SOURCE_DIR} PARENT_SCOPE)
  set(MbedOSSource_FOUND TRUE PARENT_SCOPE)
endfunction(_MbedOSSource_import)

_MbedOSSource_import()
