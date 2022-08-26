# This package is used for tensorflow lite only

function(_TensorFlowRuy_Build)
  # NOTE This line prevents multiple definitions of ruy target
  if(TARGET tensorflow-ruy)
    set(TensorFlowRuy_FOUND TRUE PARENT_SCOPE)
    return()
  endif(TARGET tensorflow-ruy)

  nnas_find_package(TensorFlowRuySource EXACT 2.3 QUIET)
  nnfw_find_package(CpuInfo QUIET)

  if(NOT TensorFlowRuySource_FOUND)
    message(STATUS "TENSORFLOWRUY: Source not found")
    set(TensorFlowRuy_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT TensorFlowRuySource_FOUND)

  if (NOT CpuInfo_FOUND)
    message(STATUS "TENSORFLOWRUY: CPUINFO not found")
    set(TensorFlowRuy_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT CpuInfo_FOUND)

  add_extdirectory("${CMAKE_CURRENT_LIST_DIR}/TensorFlowRuy" TensorFlowRuy)
  set(TensorFlowRuy_FOUND TRUE PARENT_SCOPE)
endfunction(_TensorFlowRuy_Build)

if(BUILD_TENSORFLOWRUY)
  _TensorFlowRuy_Build()
else(BUILD_TENSORFLOWRUY)
  set(TensorFlowRuy_FOUND FASLE)
endif(BUILD_TENSORFLOWRUY)
