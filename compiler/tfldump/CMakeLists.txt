if(NOT TARGET mio_tflite260)
  message(STATUS "Build tfldump: FAILED (missing mio_tflite260)")
  return()
endif(NOT TARGET mio_tflite260)

set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(tfldump ${DRIVER} ${SOURCES})
target_include_directories(tfldump PRIVATE include)
target_link_libraries(tfldump arser)
target_link_libraries(tfldump mio_tflite260)
target_link_libraries(tfldump safemain)
target_link_libraries(tfldump flatbuffers-1.12)
