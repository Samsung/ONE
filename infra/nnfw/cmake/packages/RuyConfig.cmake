function(_Ruy_Build)
  # NOTE This line prevents multiple definitions of ruy target
  if(TARGET ruy)
    set(Ruy_FOUND TRUE PARENT_SCOPE)
    return()
  endif(TARGET ruy)

  nnas_find_package(TensorFlowRuySource EXACT 2.16.1 QUIET)
  nnfw_find_package(CpuInfo QUIET)

  if(NOT TensorFlowRuySource_FOUND)
    message(STATUS "RUY: Source not found")
    set(Ruy_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT TensorFlowRuySource_FOUND)

  if (NOT CpuInfo_FOUND)
    message(STATUS "RUY: CPUINFO not found")
    set(Ruy_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT CpuInfo_FOUND)

  # Ruy's cmake requires cmake >= 3.14
  # If we ready cmake >= 3.14, enable below comment out code
  #if(PROFILE_RUY)
  #  # Will be used on ruy build
  #  set(RUY_PROFILER ON)
  #endif(PROFILE_RUY)
  #add_extdirectory("${RuySource_DIR}" Ruy)
  #
  ## Ignore warning from ruy
  #target_compile_options(ruy INTERFACE -Wno-comment)

  add_extdirectory("${CMAKE_CURRENT_LIST_DIR}/Ruy" ruy)
  set(Ruy_FOUND TRUE PARENT_SCOPE)
endfunction(_Ruy_Build)

if(BUILD_RUY)
  _Ruy_Build()
else(BUILD_RUY)
  set(Ruy_FOUND FASLE)
endif(BUILD_RUY)
