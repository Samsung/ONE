function(_GIO_2_0_import)
  nnfw_find_package(GLib2.0 REQUIRED)
  nnfw_find_package(GObject2.0 REQUIRED)

  find_library(GIO_LIBRARIES
    NAMES gio-2.0)

  # The gio-2.0 requires glib-2.0 and access the header file based on
  # the glib-2.0 include directory.
  set(GIO_INCLUDE_DIRS ${GLIB2.0_INCLUDE_DIRS} ${GOBJECT2.0_INCLUDE_DIRS})
  set(GIO_LIBRARIES ${GIO_LIBRARIES} ${GOBJECT2.0_LIBRARIES})

  set(GIO_FOUND TRUE)

  if(NOT GIO_LIBRARIES)
    set(GIO_FOUND FALSE)
  endif(NOT GIO_LIBRARIES)

  if(NOT GIO_INCLUDE_DIRS)
    set(GIO_FOUND FALSE)
  endif(NOT GIO_INCLUDE_DIRS)

  if(NOT GIO_FOUND)
    message(STATUS "Failed to find gio-2.0")
  endif(NOT GIO_FOUND)

  set(GIO2.0_FOUND ${GIO_FOUND} PARENT_SCOPE)
  set(GIO2.0_INCLUDE_DIRS ${GIO_INCLUDE_DIRS} PARENT_SCOPE)
  set(GIO2.0_LIBRARIES ${GIO_LIBRARIES} PARENT_SCOPE)
endfunction(_GIO_2_0_import)

_GIO_2_0_import()
