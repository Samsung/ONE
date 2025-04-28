function(_Nonius_import)
  nnfw_find_package(NoniusSource QUIET)

  if(NOT NoniusSource_FOUND)
    set(Nonius_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT NoniusSource_FOUND)

  if(NOT TARGET nonius)
    message(STATUS "Found nonius: TRUE")
    add_library(nonius INTERFACE)
    target_include_directories(nonius INTERFACE "${NoniusSource_DIR}/include")
  endif(NOT TARGET nonius)

  set(Nonius_FOUND TRUE PARENT_SCOPE)
endfunction(_Nonius_import)

_Nonius_import()
