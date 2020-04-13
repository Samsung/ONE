if(NOT BUILD_MINIMAL_SAMPLE)
  return()
endif(NOT BUILD_MINIMAL_SAMPLE)

list(APPEND MINIMAL_SRCS "src/minimal.cc")

add_executable(minimal ${MINIMAL_SRCS})
target_link_libraries(minimal nnfw-dev pthread dl)

install(TARGETS minimal DESTINATION bin)
