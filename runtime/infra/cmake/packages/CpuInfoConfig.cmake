function(_CpuInfo_Build)
  nnfw_find_package(CpuInfoSource QUIET)

  # NOTE This line prevents multiple definitions of cpuinfo target
  if(TARGET cpuinfo)
    set(CpuInfoSource_DIR ${CpuInfoSource_DIR} PARENT_SCOPE)
    set(CpuInfo_FOUND TRUE PARENT_SCOPE)
    return()
  endif(TARGET cpuinfo)

  if(NOT CpuInfoSource_FOUND)
    message(STATUS "CPUINFO: Source not found")
    set(CpuInfo_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT CpuInfoSource_FOUND)

  nnfw_include(ExternalProjectTools)

  # Set build option
  # - Static (position independent)
  # - No logging
  # - Library only (CPUINFO_RUNTIME_TYPE is not used)
  set(CPUINFO_LIBRARY_TYPE "static" CACHE STRING "")
  set(CPUINFO_LOG_LEVEL "none" CACHE STRING "")
  set(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "")
  set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "")
  set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "")
  set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "")
  add_extdirectory("${CpuInfoSource_DIR}" cpuinfo EXCLUDE_FROM_ALL)
  set_target_properties(cpuinfo PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set(CpuInfoSource_DIR ${CpuInfoSource_DIR} PARENT_SCOPE)
  set(CpuInfo_FOUND TRUE PARENT_SCOPE)
endfunction(_CpuInfo_Build)

if(BUILD_CPUINFO)
  _CpuInfo_Build()
else(BUILD_CPUINFO)
  set(CpuInfo_FOUND FALSE)
endif(BUILD_CPUINFO)
