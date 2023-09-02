if(NOT TARGET mio_tflite2121)
  return()
endif(NOT TARGET mio_tflite2121)

set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(tfl-inspect ${DRIVER} ${SOURCES})
target_include_directories(tfl-inspect PRIVATE src)
target_link_libraries(tfl-inspect arser)
target_link_libraries(tfl-inspect foder)
target_link_libraries(tfl-inspect mio_tflite2121)
target_link_libraries(tfl-inspect mio_tflite2121_helper)
target_link_libraries(tfl-inspect safemain)
