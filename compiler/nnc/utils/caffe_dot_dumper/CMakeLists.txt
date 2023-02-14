if(NOT TARGET mir_caffe_importer)
  return()
endif()

add_executable(caffe_model_dumper model_dump.cpp)
target_link_libraries(caffe_model_dumper PRIVATE nnc_support mir_caffe_importer)
