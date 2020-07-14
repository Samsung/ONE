if(NOT TARGET mio_tflite)
  message(STATUS "Build tfldump: FAILED (missing mio_tflite)")
  return()
endif(NOT TARGET mio_tflite)

set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(tfldump ${DRIVER} ${SOURCES})
target_include_directories(tfldump PRIVATE include)
target_link_libraries(tfldump mio_tflite)
target_link_libraries(tfldump safemain)
target_link_libraries(tfldump stdex)
target_link_libraries(tfldump flatbuffers)
