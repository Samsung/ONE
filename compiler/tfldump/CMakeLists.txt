if(NOT TARGET mio_tflite280)
  message(STATUS "Build tfldump: FAILED (missing mio_tflite280)")
  return()
endif(NOT TARGET mio_tflite280)

set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(tfldump ${DRIVER} ${SOURCES})
target_include_directories(tfldump PRIVATE include)
target_link_libraries(tfldump arser)
target_link_libraries(tfldump mio_tflite280)
target_link_libraries(tfldump mio_tflite280_helper)
target_link_libraries(tfldump safemain)
