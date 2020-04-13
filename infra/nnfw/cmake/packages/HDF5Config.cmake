unset(HDF5_DIR CACHE)
find_package(HDF5 QUIET)

if (NOT HDF5_FOUND)
  # Give second chance for some systems where sytem find_package config mode fails
  unset(HDF5_FOUND)

  find_path(HDF5_INCLUDE_DIRS NAMES hdf5.h PATH_SUFFIXES include/hdf5/serial)

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
list(APPEND HDF5_LIBRARIES "aec")
