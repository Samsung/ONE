function(_Ruy_import)
  nnfw_find_package(RuySource QUIET)

  if(NOT RuySource_FOUND)
    set(Ruy_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT RuySource_FOUND)

  set(Ruy_FOUND TRUE PARENT_SCOPE)
endfunction(_Ruy_import)

_Ruy_import()
