if(NOT TARGET mir_caffe2_importer)
  return()
endif()

add_executable(caffe2_model_dumper model_dump.cpp)
target_link_libraries(caffe2_model_dumper PRIVATE nnc_support mir_caffe2_importer)
