nnas_find_package(Caffe QUIET)

if(NOT Caffe_FOUND)
  return()
endif(NOT Caffe_FOUND)

add_subdirectory(support)
add_subdirectory(backend)
