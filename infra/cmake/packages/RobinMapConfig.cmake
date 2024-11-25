function(_RobinMap_import)
  nnas_find_package(RobinMapSource QUIET)

  if(NOT RobinMapSource_FOUND)
    set(RobinMap_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT RobinMapSource_FOUND)

  nnas_include(ExternalBuildTools)
  ExternalBuild_CMake(CMAKE_DIR   ${RobinMapSource_DIR}
                      BUILD_DIR   ${CMAKE_BINARY_DIR}/externals/ROBINMAP/build
                      INSTALL_DIR ${EXT_OVERLAY_DIR}
                      IDENTIFIER  "1.3.0")

  # execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  #                 OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  # find_path(RobinMap_INCLUDE_DIRS NAMES nanobind.h PATHS ${EXT_OVERLAY_DIR} PATH_SUFFIXES include/nanobind)
  find_package(tsl-robin-map PATHS ${EXT_OVERLAY_DIR} CONFIG REQUIRED)

  set(RobinMap_FOUND TRUE PARENT_SCOPE)
endfunction(_RobinMap_import)

_RobinMap_import()
