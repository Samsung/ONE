if(NOT TARGET mir_tflite_importer)
  return()
endif()

add_executable(tflite_model_dumper model_dump.cpp)

target_link_libraries(tflite_model_dumper PRIVATE nnc_support mir_tflite_importer)
