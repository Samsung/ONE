if(NOT TARGET mir_caffe_importer)
  return()
endif()

if(NOT TARGET mir2loco)
  return()
endif()

if(NOT TARGET exo)
  return()
endif()

message(STATUS "Build caffe2circle: TRUE")

add_executable(caffe2circle src/caffe2circle.cpp)
target_link_libraries(caffe2circle PRIVATE mir_caffe_importer mir2loco exo)
