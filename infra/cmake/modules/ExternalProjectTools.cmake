macro(add_extdirectory DIR TAG)
  # Disable warning messages from external source code
  if(IGNORE_EXTERNAL_WARNINGS)
    add_compile_options(-w)
  endif(IGNORE_EXTERNAL_WARNINGS)

  cmake_parse_arguments(ARG "EXCLUDE_FROM_ALL" "" "" ${ARGN})
  if(ARG_EXCLUDE_FROM_ALL)
    add_subdirectory(${DIR} "${CMAKE_BINARY_DIR}/externals/${TAG}" EXCLUDE_FROM_ALL)
  else(ARG_EXCLUDE_FROM_ALL)
    add_subdirectory(${DIR} "${CMAKE_BINARY_DIR}/externals/${TAG}")
  endif(ARG_EXCLUDE_FROM_ALL)
endmacro(add_extdirectory)
