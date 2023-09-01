if(NOT TARGET mio_circle06)
  message(STATUS "Skip circledump: mio_circle06 not found")
  return()
endif(NOT TARGET mio_circle06)

set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(circledump ${DRIVER} ${SOURCES})
target_include_directories(circledump PRIVATE include)
target_link_libraries(circledump arser)
target_link_libraries(circledump foder)
target_link_libraries(circledump mio_circle06)
target_link_libraries(circledump mio_circle06_helper)
target_link_libraries(circledump safemain)

install(TARGETS circledump DESTINATION bin)
