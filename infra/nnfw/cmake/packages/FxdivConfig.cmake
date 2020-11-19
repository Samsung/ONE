function(_Fxdiv_Build)
  nnas_find_package(FxdivSource QUIET)

  # NOTE This line prevents multiple definitions of target
  if(TARGET fxdiv)
    set(FxdivSource_DIR ${FxdivSource_DIR} PARENT_SCOPE)
    set(Fxdiv_FOUND TRUE PARENT_SCOPE)
    return()
  endif(TARGET fxdiv)

  if(NOT FxdivSource_FOUND)
    message(STATUS "FXDIV: Source not found")
    set(Fxdiv_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT FxdivSource_FOUND)

  set(FXDIV_BUILD_TESTS OFF CACHE BOOL "Build FXdiv unit tests")
  set(FXDIV_BUILD_BENCHMARKS OFF CACHE BOOL "Build FXdiv micro-benchmarks")

  add_extdirectory("${FxdivSource_DIR}" FXDIV EXCLUDE_FROM_ALL)
  set(FxdivSource_DIR ${FxdivSource_DIR} PARENT_SCOPE)
  set(Fxdiv_FOUND TRUE PARENT_SCOPE)
endfunction(_Fxdiv_Build)

if(BUILD_FXDIV)
  _Fxdiv_Build()
else()
  set(Fxdiv_FOUND FALSE)
endif()
