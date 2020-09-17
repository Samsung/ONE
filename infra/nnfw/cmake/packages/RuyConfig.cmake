function(_Ruy_Build)
  # NOTE This line prevents multiple definitions of ruy target
  if(TARGET ruy)
    set(Ruy_FOUND TRUE PARENT_SCOPE)
    return()
  endif(TARGET ruy)

  nnas_find_package(RuySource QUIET)
  nnfw_find_package(CpuInfo QUIET)

  if(NOT RuySource_FOUND)
    message(STATUS "RUY: Source not found")
    set(Ruy_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT RuySource_FOUND)

  if (NOT CpuInfo_FOUND)
    message(STATUS "RUY: CPUINFO not found")
    set(Ruy_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT CpuInfo_FOUND)

  add_extdirectory("${CMAKE_CURRENT_LIST_DIR}/Ruy" ruy)
  set(Ruy_FOUND TRUE PARENT_SCOPE)
endfunction(_Ruy_Build)

if(BUILD_RUY)
  _Ruy_Build()
else(BUILD_RUY)
  set(Ruy_FOUND FASLE)
endif(BUILD_RUY)
