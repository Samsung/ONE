function(_GLIB_2_0_import)
  # Find the header & lib
  find_package(PkgConfig REQUIRED)
  pkg_search_module(GLIB REQUIRED glib-2.0)

  set(GLIB_FOUND TRUE)

  if(NOT GLIB_INCLUDE_DIRS)
    set(GLIB_FOUND FALSE)
  endif(NOT GLIB_INCLUDE_DIRS)

  if(NOT GLIB_LIBRARY_DIRS AND NOT GLIB_LIBRARIES)
    set(GLIB_FOUND FALSE)
  endif(NOT GLIB_LIBRARY_DIRS AND NOT GLIB_LIBRARIES)

  if(NOT GLIB_FOUND)
    message(STATUS "Failed to find glib-2.0")
  endif(NOT GLIB_FOUND)

  set(GLIB2.0_FOUND ${GLIB_FOUND} PARENT_SCOPE)
  set(GLIB2.0_INCLUDE_DIRS ${GLIB_INCLUDE_DIRS} PARENT_SCOPE)
  set(GLIB2.0_LIBRARIES ${GLIB_LIBRARIES} PARENT_SCOPE)
endfunction(_GLIB_2_0_import)

_GLIB_2_0_import()
