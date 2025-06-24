function(_Pybind11_Build)
  nnfw_find_package(Pybind11Source QUIET)

  if(NOT Pybind11Source_FOUND)
    set(Pybind11_FOUND FALSE)
    return()
  endif(NOT Pybind11Source_FOUND)

  if(NOT TARGET pybind11)
    # FindPython Development.Module component requires cmake >= 3.18
    # 3.21.3 is cmake version in Tizen 7.0
    cmake_minimum_required(VERSION 3.21.3)

    find_package(Python COMPONENTS Interpreter)
    find_package(Python COMPONENTS Development.Module)

    nnfw_include(ExternalProjectTools)
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
