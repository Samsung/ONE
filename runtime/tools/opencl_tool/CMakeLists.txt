if (NOT BUILD_OPENCL_TOOL)
  return()
endif (NOT BUILD_OPENCL_TOOL)

list(APPEND OPENCL_INFO_SOURCE "src/opencl_info.cc")

nnfw_find_package(ARMCompute QUIET)
if(NOT ARMCompute_FOUND)
  message(WARNING "Cannot find ARM Compute Library")
  return()
endif(NOT ARMCompute_FOUND)

add_executable(opencl_info ${OPENCL_INFO_SOURCE})
target_link_libraries(opencl_info arm_compute)

install(TARGETS opencl_info DESTINATION bin)
