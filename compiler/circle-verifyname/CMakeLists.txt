if(NOT TARGET mio_circle)
  return()
endif(NOT TARGET mio_circle)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(circle_verifyname ${SOURCES})
target_include_directories(circle_verifyname PRIVATE src)
target_link_libraries(circle_verifyname arser)
target_link_libraries(circle_verifyname safemain)
target_link_libraries(circle_verifyname cwrap)
target_link_libraries(circle_verifyname foder)
target_link_libraries(circle_verifyname luci_import)
target_link_libraries(circle_verifyname luci_service)
