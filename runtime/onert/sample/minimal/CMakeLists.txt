if(NOT BUILD_MINIMAL_SAMPLE)
  return()
endif(NOT BUILD_MINIMAL_SAMPLE)

list(APPEND MINIMAL_SRCS "src/minimal.cc")

add_executable(onert-minimal-app ${MINIMAL_SRCS})
target_link_libraries(onert-minimal-app nnfw-dev pthread dl)

install(TARGETS onert-minimal-app DESTINATION bin)
