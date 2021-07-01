function(_Opencl_Headers_import)
  nnas_find_package(Opencl_HeadersSource QUIET)

  # NOTE This line prevents multiple definitions of target
  if(TARGET Headers)
    set(Opencl_HeadersSource_DIR ${Opencl_HeadersSource_DIR} PARENT_SCOPE)
    set(Opencl_Headers_FOUND TRUE PARENT_SCOPE)
    return()
  endif(TARGET Headers)

  if(NOT Opencl_HeadersSource_FOUND)
    message(STATUS "Opencl_Headers: Source not found")
    set(Opencl_Headers_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT Opencl_HeadersSource_FOUND)

  add_extdirectory("${Opencl_HeadersSource_DIR}" OPENCL_HEADERS EXCLUDE_FROM_ALL)
  set(Opencl_Headers_DIR ${Opencl_HeadersSource_DIR} PARENT_SCOPE)
  set(Opencl_Headers_FOUND TRUE PARENT_SCOPE)
endfunction(_Opencl_Headers_import)

_Opencl_Headers_import()
