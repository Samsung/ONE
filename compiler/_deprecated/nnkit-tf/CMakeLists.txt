nnas_find_package(TensorFlow EXACT 1.13 QUIET)

if(NOT TensorFlow_FOUND)
  return()
endif(NOT TensorFlow_FOUND)

add_subdirectory(support)
add_subdirectory(backend)
