if(NOT TARGET mio_circle07)
  return()
endif(NOT TARGET mio_circle07)

set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(circle-operator ${DRIVER} ${SOURCES})
target_include_directories(circle-operator PRIVATE src)
target_link_libraries(circle-operator arser)
target_link_libraries(circle-operator foder)
target_link_libraries(circle-operator mio_circle07)
target_link_libraries(circle-operator mio_circle07_helper)
target_link_libraries(circle-operator safemain)

install(TARGETS circle-operator DESTINATION bin)
