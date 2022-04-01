#if(NOT NPU_CHECK)
#  return()
#endif(NOT NPU_CHECK)

nnfw_find_package(NPUEngine QUIET 2.3.0)

if(NOT NPUEngine_FOUND)
  message(FATAL_ERROR "Find NPUEngine devel package: Cannot found")
  return()
else(NOT NPUEngine_FOUND)
  message(STATUS "Find NPUEngine: Found")
endif(NOT NPUEngine_FOUND)
