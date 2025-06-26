function(_HDF5_build)
  if(NOT BUILD_HDF5)
    return()
  endif(NOT BUILD_HDF5)

  nnfw_find_package(HDF5Source QUIET)

  if(NOT HDF5Source_FOUND)
    message(STATUS "HD5Config skip: HDF5Source NOT FOUND")
    return()
  endif(NOT HDF5Source_FOUND)

  nnfw_include(ExternalBuildTools)
  ExternalBuild_CMake(CMAKE_DIR   ${HDF5Source_DIR}
                      BUILD_DIR   ${CMAKE_BINARY_DIR}/externals/HDF5
                      INSTALL_DIR ${EXT_OVERLAY_DIR}
                      IDENTIFIER  "1.14.6"
                      PKG_NAME    "HDF5"
                      EXTRA_OPTS "-DBUILD_SHARED_LIBS:BOOL=OFF"
                                 "-DBUILD_TESTING:BOOL=OFF"
                                 "-DHDF5_BUILD_CPP_LIB:BOOL=ON"
                                 "-DHDF5_BUILD_EXAMPLES:BOOL=OFF"
                                 "-DHDF5_BUILD_TOOLS:BOOL=OFF"
                                 "-DHDF5_ENABLE_SZIP_SUPPORT:BOOL=OFF"
                                 "-DHDF5_MODULE_MODE_ZLIB:BOOL=OFF"
                                 "-DHDF5_ENABLE_Z_LIB_SUPPORT:BOOL=OFF"
                                 "-DHDF5_BUILD_UTILS:BOOL=OFF")

endfunction(_HDF5_build)

if (NOT HDF5_FOUND AND BUILD_HDF5)
  _HDF5_build()

  find_package(hdf5 REQUIRED COMPONENTS static C CXX)

  # Set HDF5_CXX_STATIC_LIBRARY manually
  # hdf5 cmake config file fails to set HDF5_CXX_STATIC_LIBRARY
  if (NOT HDF5_CXX_STATIC_LIBRARY)
    set(HDF5_CXX_STATIC_LIBRARY hdf5_cpp-static)
  endif()
endif()
