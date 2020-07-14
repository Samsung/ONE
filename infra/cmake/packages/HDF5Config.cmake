function(_HDF5_build)
  if(NOT BUILD_HDF5)
    return()
  endif(NOT BUILD_HDF5)

  nnas_find_package(HDF5Source QUIET)

  if(NOT HDF5Source_FOUND)
    return()
  endif(NOT HDF5Source_FOUND)

  nnas_include(ExternalBuildTools)
  ExternalBuild_CMake(CMAKE_DIR   ${HDF5Source_DIR}
                      BUILD_DIR   ${CMAKE_BINARY_DIR}/externals/HDF5/build
                      INSTALL_DIR ${EXT_OVERLAY_DIR}
                      IDENTIFIER  "1.8.16"
                      PKG_NAME    "HDF5"
                      EXTRA_OPTS "-DBUILD_SHARED_LIBS:BOOL=ON"
                                 "-DHDF5_BUILD_TOOLS:BOOL=ON"
                                 "-DHDF5_ENABLE_SZIP_SUPPORT:BOOL=OFF"
                                 "-DHDF5_ENABLE_Z_LIB_SUPPORT:BOOL=OFF")

endfunction(_HDF5_build)

_HDF5_build()

find_path(HDF5_CONFIG_DIR "hdf5-config.cmake"
          PATHS ${EXT_OVERLAY_DIR}
          PATH_SUFFIXES
            share/cmake
            share/cmake/hdf5
            cmake/hdf5
            lib/cmake/hdf5)

include(${HDF5_CONFIG_DIR}/hdf5-config.cmake)

unset(HDF5_INCLUDE_DIRS)
unset(HDF5_C_INCLUDE_DIRS)
unset(HDF5_CXX_INCLUDE_DIRS)
unset(HDF5_HL_INCLUDE_DIRS)

unset(HDF5_LIBRARIES)
unset(HDF5_HL_LIBRARIES)
unset(HDF5_C_LIBRARIES)
unset(HDF5_CXX_LIBRARIES)
unset(HDF5_C_HL_LIBRARIES)
unset(HDF5_CXX_HL_LIBRARIES)

# If user doesn't specify static or shared, set it to shared by default
list(FIND HDF5_FIND_COMPONENTS "STATIC" _index)
if(${_index} GREATER -1)
  # static
  set(_SUFFIX "-static")
else()
  # shared
  set(_SUFFIX "-shared")
endif()

list(REMOVE_ITEM HDF5_FIND_COMPONENTS "static;shared")
set(HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIR})
foreach(COMP HDF5_FIND_COMPONENTS)
  set(HDF5_${COMP}_INCLUDE_DIRS ${HDF5_INCLUDE_DIR})
endforeach()

set(HDF5_LIBRARIES "hdf5${_SUFFIX}")
set(HDF5_C_LIBRARIES "hdf5${_SUFFIX}")
set(HDF5_CXX_LIBRARIES "hdf5_cpp${_SUFFIX}")
set(HDF5_HL_LIBRARIES "hdf5_hl${_SUFFIX}")
set(HDF5_C_HL_LIBRARIES "hdf5_hl${_SUFFIX}")
set(HDF5_CXX_HL_LIBRARIES "hdf5_hl_cpp${_SUFFIX}")
