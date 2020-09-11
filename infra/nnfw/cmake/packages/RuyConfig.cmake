function(_Ruy_import)
  # NOTE This line prevents multiple definitions of ruy target
  if(TARGET ruy)
    set(Ruy_FOUND TRUE)
    return()
  endif(TARGET ruy)

  nnas_find_package(RuySource QUIET)

  if(NOT RuySource_FOUND)
    set(Ruy_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT RuySource_FOUND)

  if(BUILD_RUY)
    add_extdirectory("${CMAKE_CURRENT_LIST_DIR}/Ruy" ruy)
  endif(BUILD_RUY)

  set(Ruy_FOUND TRUE PARENT_SCOPE)
endfunction(_Ruy_import)

_Ruy_import()
