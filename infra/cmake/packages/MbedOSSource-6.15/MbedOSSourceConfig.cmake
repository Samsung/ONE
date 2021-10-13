function(_MbedOSSource_import)
  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(MBEDOS_6_15_URL https://github.com/ARMmbed/mbed-os/archive/refs/tags/mbed-os-6.15.0.tar.gz)
  set(MBEDOS_6_15_SHA256 529b04c41f3020ed8a62f12d47f2d3de87e1b07fb13708534534a587f7ea048e)

  ExternalSource_Download(MBEDOS DIRNAME MBEDOS-6.15 ${MBEDOS_6_15_URL}
          CHECKSUM "SHA256=${MBEDOS_6_15_SHA256}")

  set(MbedOSSource_DIR ${MBEDOS_SOURCE_DIR} PARENT_SCOPE)
  set(MbedOSSource_FOUND TRUE PARENT_SCOPE)
endfunction(_MbedOSSource_import)

_MbedOSSource_import()
