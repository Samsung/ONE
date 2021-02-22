if(NOT TARGET mio_tf)
  return()
endif(NOT TARGET mio_tf)

message(STATUS "Build tfkit: TRUE")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(tfkit ${SOURCES})
target_link_libraries(tfkit PRIVATE cli)
target_link_libraries(tfkit PRIVATE mio_tf)
target_link_libraries(tfkit PRIVATE nncc_common)
