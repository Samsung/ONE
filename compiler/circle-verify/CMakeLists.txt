if(NOT TARGET mio_circle05)
  message(STATUS "Skip circle-verify: mio_circle05 not found")
  return()
endif(NOT TARGET mio_circle05)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(circle-verify ${SOURCES})
target_include_directories(circle-verify PRIVATE src)
target_link_libraries(circle-verify arser)
target_link_libraries(circle-verify mio_circle05)
target_link_libraries(circle-verify safemain)
target_link_libraries(circle-verify cwrap)
target_link_libraries(circle-verify foder)
