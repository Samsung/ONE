if(NOT TARGET mio_circle)
  return()
endif(NOT TARGET mio_circle)

set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(circledump ${DRIVER} ${SOURCES})
target_include_directories(circledump PRIVATE include)
target_link_libraries(circledump arser)
target_link_libraries(circledump mio_circle)
target_link_libraries(circledump safemain)
target_link_libraries(circledump flatbuffers)
