function(_Opencl_HeadersSource_import)
  if(NOT DOWNLOAD_OPENCL_HEADERS)
    set(Opencl_HeadersSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_OPENCL_HEADERS)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(OPENCL_HEADERS_URL ${EXTERNAL_DOWNLOAD_SERVER}/KhronosGroup/OpenCL-Headers/archive/v2021.04.29.tar.gz)

  ExternalSource_Download(OPENCL_HEADERS
    DIRNAME OPENCL_HEADERS
    URL ${OPENCL_HEADERS_URL}
    CHECKSUM MD5=5a7ea04265119aa76b4ecbd95f258219)

  set(Opencl_HeadersSource_DIR ${OPENCL_HEADERS_SOURCE_DIR} PARENT_SCOPE)
  set(Opencl_HeadersSource_FOUND TRUE PARENT_SCOPE)
endfunction(_Opencl_HeadersSource_import)

_Opencl_HeadersSource_import()
