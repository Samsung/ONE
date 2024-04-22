function(_GIO_UNIX_2_0_import)
  nnfw_find_package(Gio2.0 REQUIRED)

  find_path(GIO_UNIX_INCLUDE_DIR
    NAMES gio/gfiledescriptorbased.h
    PATH_SUFFIXES gio-unix-2.0)

  # The gio-unix-2.0 requires gio-2.0 and link the gio-2.0 library.
  set(GIO_UNIX_LIBRARIES ${GIO2.0_LIBRARIES})

  set(GIO_UNIX_FOUND TRUE)

  if(NOT GIO_UNIX_LIBRARIES)
    set(GIO_UNIX_FOUND FALSE)
  endif(NOT GIO_UNIX_LIBRARIES)

  if(NOT GIO_UNIX_INCLUDE_DIR)
    set(GIO_UNIX_FOUND FALSE)
  endif(NOT GIO_UNIX_INCLUDE_DIR)

  if(NOT GIO_UNIX_FOUND)
    message(STATUS "Failed to find gio-unix-2.0")
  endif(NOT GIO_UNIX_FOUND)

  set(GIO_UNIX_2.0_FOUND ${GIO_UNIX_FOUND} PARENT_SCOPE)
  set(GIO_UNIX_2.0_INCLUDE_DIRS ${GIO_UNIX_INCLUDE_DIR} PARENT_SCOPE)
  set(GIO_UNIX_2.0_LIBRARIES ${GIO_UNIX_LIBRARIES} PARENT_SCOPE)
endfunction(_GIO_UNIX_2_0_import)

_GIO_UNIX_2_0_import()
