function(_Opencl_Headers_import)
  nnas_find_package(Opencl_HeadersSource QUIET)

  if(NOT Opencl_HeadersSource_FOUND)
    set(Opencl_Headers_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT Opencl_HeadersSource_FOUND)

  if(NOT TARGET opencl_headers)
    nnas_include(ExternalProjectTools)
    add_extdirectory("${Opencl_HeadersSource_DIR}" OPENCL_HEADERS)
  endif(NOT TARGET opencl_headers)

  set(Opencl_Headers_FOUND TRUE PARENT_SCOPE)
endfunction(_Opencl_Headers_import)

_Opencl_Headers_import()
