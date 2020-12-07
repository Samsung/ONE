function(_Pthreadpool_Build)
  nnas_find_package(PthreadpoolSource QUIET)

  # NOTE This line prevents multiple definitions of target
  if(TARGET pthreadpool)
    set(PthreadpoolSource_DIR ${PthreadpoolSource_DIR} PARENT_SCOPE)
    set(Pthreadpool_FOUND TRUE PARENT_SCOPE)
    return()
  endif(TARGET pthreadpool)

  if(NOT PthreadpoolSource_FOUND)
    message(STATUS "PTHREADPOOL: Source not found")
    set(Pthreadpool_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT PthreadpoolSource_FOUND)

  SET(PTHREADPOOL_BUILD_TESTS OFF CACHE BOOL "Build pthreadpool unit tests")
  SET(PTHREADPOOL_BUILD_BENCHMARKS OFF CACHE BOOL "Build pthreadpool micro-benchmarks")

  nnas_find_package(FxdivSource)
  set(FXDIV_SOURCE_DIR ${FxdivSource_DIR} CACHE STRING "String to disable download FXDIV")

  add_extdirectory("${PthreadpoolSource_DIR}" PTHREADPOOL EXCLUDE_FROM_ALL)
  set_target_properties(pthreadpool PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set(PthreadpoolSource_DIR ${PthreadpoolSource_DIR} PARENT_SCOPE)
  set(Pthreadpool_FOUND TRUE PARENT_SCOPE)
endfunction(_Pthreadpool_Build)

if(BUILD_PTHREADPOOL)
  _Pthreadpool_Build()
else()
  set(Pthreadpool_FOUND FALSE)
endif()
