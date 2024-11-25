function(_Nanobind_import)
  nnas_find_package(NanobindSource QUIET)

  if(NOT NanobindSource_FOUND)
    set(Nanobind_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT NanobindSource_FOUND)

  nnas_include(ExternalBuildTools)
  ExternalBuild_CMake(CMAKE_DIR   ${NanobindSource_DIR}
                      BUILD_DIR   ${CMAKE_BINARY_DIR}/externals/NANOBIND/build
                      INSTALL_DIR ${EXT_OVERLAY_DIR}
                      IDENTIFIER  "2.2.0"
                      PKG_NAME    "NANOBIND"
                      EXTRA_OPTS "-DNB_TEST:BOOL=OFF"
                                 "-DNB_USE_SUBMODULE_DEPS=OFF")

  # execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  #                 OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  # find_path(Nanobind_INCLUDE_DIRS NAMES nanobind.h PATHS ${EXT_OVERLAY_DIR} PATH_SUFFIXES include/nanobind)
  find_package(nanobind PATHS ${EXT_OVERLAY_DIR} CONFIG REQUIRED)

  set(Nanobind_FOUND TRUE PARENT_SCOPE)
endfunction(_Nanobind_import)

_Nanobind_import()
