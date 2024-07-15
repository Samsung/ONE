if(NOT TARGET mio_circle09)
  return()
endif(NOT TARGET mio_circle09)

set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(circle-inspect ${DRIVER} ${SOURCES})
target_include_directories(circle-inspect PRIVATE src)
target_link_libraries(circle-inspect arser)
target_link_libraries(circle-inspect foder)
target_link_libraries(circle-inspect mio_circle09)
target_link_libraries(circle-inspect mio_circle09_helper)
target_link_libraries(circle-inspect safemain)
