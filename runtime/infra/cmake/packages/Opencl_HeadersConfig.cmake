function(_Opencl_Headers_import)
  nnfw_find_package(Opencl_HeadersSource QUIET)

  # NOTE This line prevents multiple definitions of target
  if(TARGET OpenCL_Headers)
    set(Opencl_HeadersSource_DIR ${Opencl_HeadersSource_DIR} PARENT_SCOPE)
    set(Opencl_Headers_FOUND TRUE PARENT_SCOPE)
    return()
  endif(TARGET OpenCL_Headers)

  if(NOT Opencl_HeadersSource_FOUND)
    message(STATUS "Opencl_Headers: Source not found")
    set(Opencl_Headers_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT Opencl_HeadersSource_FOUND)

  # We don't need test builds and installs, we only need headers.
  # add_extdirectory("${Opencl_HeadersSource_DIR}" OPENCL_HEADERS EXCLUDE_FROM_ALL)

  add_library(OpenCL_Headers INTERFACE)
  target_include_directories(OpenCL_Headers INTERFACE ${Opencl_HeadersSource_DIR})

  set(Opencl_Headers_DIR ${Opencl_HeadersSource_DIR} PARENT_SCOPE)
  set(Opencl_Headers_FOUND TRUE PARENT_SCOPE)
endfunction(_Opencl_Headers_import)

_Opencl_Headers_import()
