nnas_find_package(Protobuf QUIET)

if(NOT Protobuf_FOUND)
  return()
endif(NOT Protobuf_FOUND)

if(NOT TARGET mio_circle)
  return()
endif(NOT TARGET mio_circle)

# Recipe Parser
add_subdirectory(proto)
# Log
add_subdirectory(log)
# Core Library
add_subdirectory(core)
# Circle Library
add_subdirectory(circle)
# Tools
add_subdirectory(tools)
if(ENABLE_TEST)
  add_subdirectory(tests)
endif(ENABLE_TEST)
