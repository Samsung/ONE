if(NOT TARGET nnkit_support_cmdline)
  return()
endif(NOT TARGET nnkit_support_cmdline)

if(NOT TARGET nnkit_support_backend)
  return()
endif(NOT TARGET nnkit_support_backend)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(nnkit-benchmark ${SOURCES})
target_link_libraries(nnkit-benchmark nnkit_support_cmdline)
target_link_libraries(nnkit-benchmark nnkit_support_backend)
