nnas_find_package(ONNXRuntime QUIET)

if(NOT ONNXRuntime_FOUND)
  return()
endif(NOT ONNXRuntime_FOUND)

add_subdirectory(support)
add_subdirectory(backend)
