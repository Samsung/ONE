if(NOT TARGET mio_circle)
  return()
endif(NOT TARGET mio_circle)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(circle-verify ${SOURCES})
target_include_directories(circle-verify PRIVATE src)
target_link_libraries(circle-verify arser)
target_link_libraries(circle-verify mio_circle)
target_link_libraries(circle-verify safemain)
target_link_libraries(circle-verify cwrap)
target_link_libraries(circle-verify foder)
