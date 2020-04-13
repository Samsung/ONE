nnas_find_package(TensorFlow QUIET)

if(NOT TensorFlow_FOUND)
  return()
endif(NOT TensorFlow_FOUND)

add_subdirectory(support)
add_subdirectory(backend)
