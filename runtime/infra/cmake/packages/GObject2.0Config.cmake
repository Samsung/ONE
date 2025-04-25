function(_GOBJECT_2_0_import)
  nnfw_find_package(GLib2.0 REQUIRED)

  find_library(GOBJECT_LIBRARIES
    NAMES gobject-2.0)

  # The gobject-2.0 requires glib-2.0 and access the header file based on
  # the glib-2.0 include directory.
  set(GOBJECT_INCLUDE_DIRS ${GLIB2.0_INCLUDE_DIRS})

  set(GOBJECT_FOUND TRUE)

  if(NOT GOBJECT_LIBRARIES)
    set(GOBJECT_FOUND FALSE)
  endif(NOT GOBJECT_LIBRARIES)

  if(NOT GOBJECT_INCLUDE_DIRS)
    set(GOBJECT_FOUND FALSE)
  endif(NOT GOBJECT_INCLUDE_DIRS)

  if(NOT GOBJECT_FOUND)
    message(STATUS "Failed to find gobject-2.0")
  endif(NOT GOBJECT_FOUND)

  set(GOBJECT2.0_FOUND ${GOBJECT_FOUND} PARENT_SCOPE)
  set(GOBJECT2.0_INCLUDE_DIRS ${GOBJECT_INCLUDE_DIRS} PARENT_SCOPE)
  set(GOBJECT2.0_LIBRARIES ${GOBJECT_LIBRARIES} PARENT_SCOPE)
endfunction(_GOBJECT_2_0_import)

_GOBJECT_2_0_import()
