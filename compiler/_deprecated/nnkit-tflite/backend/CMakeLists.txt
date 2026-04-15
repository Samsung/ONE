if(NOT TARGET nnkit_support_tflite)
  return()
endif(NOT TARGET nnkit_support_tflite)

add_library(nnkit_tflite_backend SHARED Backend.cpp)
target_link_libraries(nnkit_tflite_backend nnkit_support_tflite)
