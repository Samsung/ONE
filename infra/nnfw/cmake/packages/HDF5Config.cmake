# Don't cache HDF5_*. Otherwise it will use the cached value without searching.
unset(HDF5_DIR CACHE)
unset(HDF5_INCLUDE_DIRS CACHE)
unset(HDF5_CXX_LIBRARY_hdf5 CACHE)
unset(HDF5_CXX_LIBRARY_hdf5_cpp CACHE)

if(NOT BUILD_WITH_HDF5)
  set(HDF5_FOUND FALSE)
  return()
endif(NOT BUILD_WITH_HDF5)

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

# Case 2. search pre-installed locations (by apt, brew, ...)
if(NOT CMAKE_CROSSCOMPILING)
  find_package(HDF5 COMPONENTS CXX QUIET)
else()
  find_path(HDF5_INCLUDE_DIRS NAMES hdf5.h ONLY_CMAKE_FIND_ROOT_PATH PATH_SUFFIXES include/hdf5/serial)

  if (NOT HDF5_INCLUDE_DIRS)
    set(HDF5_FOUND FALSE)
    return()
  endif()

  if(HDF5_USE_STATIC_LIBRARIES)
    find_library(HDF5_CXX_LIBRARY_hdf5
      NAMES libhdf5.a
      ONLY_CMAKE_FIND_ROOT_PATH
      PATH_SUFFIXES hdf5/serial)
    find_library(HDF5_CXX_LIBRARY_hdf5_cpp
      NAMES libhdf5_cpp.a
      ONLY_CMAKE_FIND_ROOT_PATH
      PATH_SUFFIXES hdf5/serial)
  else(HDF5_USE_STATIC_LIBRARIES)
    find_library(HDF5_CXX_LIBRARY_hdf5
      NAMES libhdf5.so
      ONLY_CMAKE_FIND_ROOT_PATH
      PATH_SUFFIXES hdf5/serial)
    find_library(HDF5_CXX_LIBRARY_hdf5_cpp
      NAMES libhdf5_cpp.so
      ONLY_CMAKE_FIND_ROOT_PATH
      PATH_SUFFIXES hdf5/serial)
  endif(HDF5_USE_STATIC_LIBRARIES)

  if (NOT (HDF5_CXX_LIBRARY_hdf5 AND HDF5_CXX_LIBRARY_hdf5_cpp))
    set(HDF5_FOUND FALSE)
    return()
  endif()

  # We can use "hdf5" and "hdf5_cpp" to use the same file founded with above.
  list(APPEND HDF5_CXX_LIBRARIES ${HDF5_CXX_LIBRARY_hdf5} ${HDF5_CXX_LIBRARY_hdf5_cpp} "sz" "z" "dl" "m")

  # Append missing libaec which is required by libsz, which is required by libhdf5
  list(APPEND HDF5_CXX_LIBRARIES "aec")

  set(HDF5_FOUND TRUE)
endif()
