macro(add_extdirectory DIR TAG)
  cmake_parse_arguments(ARG "EXCLUDE_FROM_ALL" "" "" ${ARGN})

  # Disable warning messages from external source code
  if(DISABLE_EXTERNAL_WARNING)
    add_compile_options(-w)
  endif(DISABLE_EXTERNAL_WARNING)

  if(ARG_EXCLUDE_FROM_ALL)
    add_subdirectory(${DIR} "${CMAKE_BINARY_DIR}/externals/${TAG}" EXCLUDE_FROM_ALL)
  else(ARG_EXCLUDE_FROM_ALL)
    add_subdirectory(${DIR} "${CMAKE_BINARY_DIR}/externals/${TAG}")
  endif(ARG_EXCLUDE_FROM_ALL)
endmacro(add_extdirectory)
