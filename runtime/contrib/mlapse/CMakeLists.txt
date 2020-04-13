if(NOT BUILD_MLAPSE)
  return()
endif(NOT BUILD_MLAPSE)

message(STATUS "Build mlapse: TRUE")

# TODO Add "core"
add_subdirectory(tfl)
