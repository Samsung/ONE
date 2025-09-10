if(NOT TARGET mio_circle)
  message(STATUS "Skip circledump: mio_circle not found")
  return()
endif(NOT TARGET mio_circle)

set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(circledump ${DRIVER} ${SOURCES})
target_include_directories(circledump PRIVATE include)
target_link_libraries(circledump arser)
target_link_libraries(circledump foder)
target_link_libraries(circledump mio_circle)
target_link_libraries(circledump mio_circle_helper)
target_link_libraries(circledump safemain)

install(TARGETS circledump DESTINATION bin)
