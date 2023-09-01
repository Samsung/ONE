if(NOT TARGET mio_tflite2121)
  message(STATUS "Build tfldump: FAILED (missing mio_tflite2121)")
  return()
endif(NOT TARGET mio_tflite2121)

set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(tfldump ${DRIVER} ${SOURCES})
target_include_directories(tfldump PRIVATE include)
target_link_libraries(tfldump arser)
target_link_libraries(tfldump foder)
target_link_libraries(tfldump mio_tflite2121)
target_link_libraries(tfldump mio_tflite2121_helper)
target_link_libraries(tfldump safemain)
