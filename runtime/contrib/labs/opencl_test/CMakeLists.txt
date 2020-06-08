if(NOT ${TARGET_ARCH_BASE} STREQUAL "arm")
  return()
endif(NOT ${TARGET_ARCH_BASE} STREQUAL "arm")

list(APPEND OPENCL_INFO_SOURCE "src/opencl_test.cc")

nnfw_find_package(ARMCompute REQUIRED)

add_executable(opencl_test ${OPENCL_INFO_SOURCE})
target_link_libraries(opencl_test arm_compute)
target_link_libraries(opencl_test arm_compute_ex)
