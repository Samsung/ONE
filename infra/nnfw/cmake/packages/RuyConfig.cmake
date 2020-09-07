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
    set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "Build cpuinfo unit tests")
    set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "Build cpuinfo mock tests")
    set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "Build cpuinfo micro-benchmarks")
    add_extdirectory("${CpuInfoSource_DIR}" cpuinfo)
    set_target_properties(cpuinfo PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_extdirectory("${CMAKE_CURRENT_LIST_DIR}/Ruy" ruy)
  endif(BUILD_RUY)

  set(Ruy_FOUND TRUE PARENT_SCOPE)
endfunction(_Ruy_import)

_Ruy_import()
