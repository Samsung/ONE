function(_HDF5_build)
  if(NOT BUILD_HDF5)
    return()
  endif(NOT BUILD_HDF5)

  nnas_find_package(HDF5Source QUIET)

  if(NOT HDF5Source_FOUND)
    message(STATUS "HD5Config skip: HDF5Source NOT FOUND")
    return()
  endif(NOT HDF5Source_FOUND)

  if(DEFINED ENV{BUILD_HOST_EXEC})
    set(EXTERNAL_H5MAKE_LIBSETTINGS $ENV{BUILD_HOST_EXEC}/externals/HDF5/build/bin/H5make_libsettings)
    set(ENV{EXTERNAL_H5MAKE_LIBSETTINGS} ${EXTERNAL_H5MAKE_LIBSETTINGS})

    # NOTE https://github.com/Samsung/ONE/issues/8762
    # TODO generalize to select 'linux-armv7l'
    set(H5TINIT_C_FROM_NATIVE ${CMAKE_CURRENT_LIST_DIR}/H5Tinit.c.linux-armv7l)
    set(H5TINIT_C_COPY ${CMAKE_BINARY_DIR}/externals/HDF5/build/H5Tinit.c)
    message(STATUS "Copy H5Tinit.c generated from target native build")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E copy "${H5TINIT_C_FROM_NATIVE}" "${H5TINIT_C_COPY}"
    )
  endif(DEFINED ENV{BUILD_HOST_EXEC})

  nnas_include(ExternalBuildTools)
  ExternalBuild_CMake(CMAKE_DIR   ${HDF5Source_DIR}
                      BUILD_DIR   ${CMAKE_BINARY_DIR}/externals/HDF5/build
                      INSTALL_DIR ${EXT_OVERLAY_DIR}
                      IDENTIFIER  "1.8.16"
                      PKG_NAME    "HDF5"
                      EXTRA_OPTS "-DBUILD_SHARED_LIBS:BOOL=ON"
                                 "-DBUILD_TESTING:BOOL=OFF"
                                 "-DHDF5_BUILD_EXAMPLES:BOOL=OFF"
                                 "-DHDF5_BUILD_TOOLS:BOOL=ON"
                                 "-DHDF5_ENABLE_SZIP_SUPPORT:BOOL=OFF"
                                 "-DHDF5_ENABLE_Z_LIB_SUPPORT:BOOL=OFF"
                                 "-DCMAKE_POLICY_VERSION_MINIMUM=3.5")

endfunction(_HDF5_build)

_HDF5_build()

find_path(HDF5_CONFIG_DIR "hdf5-config.cmake"
          PATHS ${EXT_OVERLAY_DIR}
          NO_CMAKE_FIND_ROOT_PATH
          PATH_SUFFIXES
            cmake
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
