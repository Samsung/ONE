# Don't cache HDF5_*. Otherwise it will use the cached value without searching.
unset(HDF5_DIR CACHE)
unset(HDF5_INCLUDE_DIRS CACHE)
unset(HDF5_CXX_LIBRARY_hdf5 CACHE)
unset(HDF5_CXX_LIBRARY_hdf5_cpp CACHE)

# Case 1. external hdf5
if(DEFINED EXT_HDF5_DIR)
  find_path(HDF5_INCLUDE_DIRS NAMES H5Cpp.h NO_CMAKE_FIND_ROOT_PATH PATHS "${EXT_HDF5_DIR}/include")
  find_library(HDF5_CXX_LIBRARY_hdf5 NAMES libhdf5.a PATHS "${EXT_HDF5_DIR}/lib")
  find_library(HDF5_CXX_LIBRARY_hdf5_cpp NAMES libhdf5_cpp.a PATHS "${EXT_HDF5_DIR}/lib")
  if (NOT (HDF5_INCLUDE_DIRS AND HDF5_CXX_LIBRARY_hdf5 AND HDF5_CXX_LIBRARY_hdf5_cpp))
    message(WARNING "Failed to find H5Cpp.h or libhdf5.a or libhdf5_cpp.a")
    set(HDF5_FOUND FALSE)
    return()
  else()
    # message(FATAL_ERROR "0=${HDF5_INCLUDE_DIRS},1=${HDF5_CXX_LIBRARIES}")
    set(HDF5_FOUND TRUE)
    list(APPEND HDF5_CXX_LIBRARIES ${HDF5_CXX_LIBRARY_hdf5_cpp} ${HDF5_CXX_LIBRARY_hdf5})
    return()
  endif()
endif()

# Case 2. search default locations (e.g. system root, ...) for hdf5
find_package(HDF5 COMPONENTS CXX QUIET)
if (NOT HDF5_FOUND)
  # Give second chance for some systems where sytem find_package config mode fails
  unset(HDF5_FOUND)

  find_path(HDF5_INCLUDE_DIRS NAMES hdf5.h ONLY_CMAKE_FIND_ROOT_PATH PATH_SUFFIXES include/hdf5/serial)

  if (NOT HDF5_INCLUDE_DIRS)
    set(HDF5_FOUND FALSE)
    return()
  endif()

  if (HDF5_USE_STATIC_LIBRARIES)
    find_library(HDF5_LIBRARIES libhdf5.a)
  else (HDF5_USE_STATIC_LIBRARIES)
    find_library(HDF5_LIBRARIES libhdf5.so)
  endif(HDF5_USE_STATIC_LIBRARIES)

  if (NOT HDF5_LIBRARIES)
    set(HDF5_FOUND FALSE)
    return()
  endif()
  list(APPEND HDF5_LIBRARIES "sz" "z" "dl" "m")

  set(HDF5_FOUND TRUE)
endif()

# Append missing libaec which is required by libsz, which is required by libhdf5
list(APPEND HDF5_CXX_LIBRARIES "aec")
