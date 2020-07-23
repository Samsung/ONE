function(_Pybind11_import)
  nnas_find_package(Pybind11Source QUIET)

  if(NOT Pybind11Source_FOUND)
    set(Pybind11_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT Pybind11Source_FOUND)

  nnas_include(ExternalBuildTools)
  ExternalBuild_CMake(CMAKE_DIR   ${Pybind11Source_DIR}
                      BUILD_DIR   ${CMAKE_BINARY_DIR}/externals/PYBIND11/build
                      INSTALL_DIR ${EXT_OVERLAY_DIR}
                      IDENTIFIER  "2.3.0"
                      PKG_NAME    "PYBIND11")

  find_path(Pybind11_INCLUDE_DIRS NAMES pybind11.h PATHS ${EXT_OVERLAY_DIR} PATH_SUFFIXES include/pybind11)

  set(Pybind11_FOUND TRUE PARENT_SCOPE)
endfunction(_Pybind11_import)

_Pybind11_import()
