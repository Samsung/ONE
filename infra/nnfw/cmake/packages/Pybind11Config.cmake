function(_Pybind11_Build)
  nnas_find_package(Pybind11Source QUIET)

  if(NOT Pybind11Source_FOUND)
    set(Pybind11_FOUND FALSE)
    return()
  endif(NOT Pybind11Source_FOUND)

  if(NOT TARGET pybind11)
    nnas_include(ExternalProjectTools)
    add_extdirectory(${Pybind11Source_DIR} pybind11 EXCLUDE_FROM_ALL)
  endif(NOT TARGET pybind11)

  set(Pybind11_FOUND TRUE)
  return()
endfunction(_Pybind11_Build)

if(BUILD_PYTHON_BINDING)
  _Pybind11_Build()
else()
  set(Pybind11_FOUND FALSE)
endif()
