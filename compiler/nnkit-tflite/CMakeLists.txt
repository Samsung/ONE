nnas_find_package(TensorFlowLite QUIET EXACT 1.13.1)

if(NOT TensorFlowLite_FOUND)
  return()
endif(NOT TensorFlowLite_FOUND)

add_subdirectory(support)
add_subdirectory(backend)
