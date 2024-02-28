if(NOT TARGET mio_circle08)
  message(STATUS "Skip circle-verify: mio_circle08 not found")
  return()
endif(NOT TARGET mio_circle08)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(circle-verify ${SOURCES})
target_include_directories(circle-verify PRIVATE src)
target_link_libraries(circle-verify arser)
target_link_libraries(circle-verify mio_circle08)
target_link_libraries(circle-verify safemain)
target_link_libraries(circle-verify cwrap)
target_link_libraries(circle-verify foder)
