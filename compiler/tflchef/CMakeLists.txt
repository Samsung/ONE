nnas_find_package(Protobuf QUIET)

if(NOT Protobuf_FOUND)
  message(STATUS "Build tflchef: FAILED (missing Protobuf)")
  return()
endif(NOT Protobuf_FOUND)

if(NOT TARGET mio_tflite)
  message(STATUS "Build tflchef: FAILED (missing mio_tflite)")
  return()
endif(NOT TARGET mio_tflite)

# Recipe Parser
add_subdirectory(proto)
# Log
add_subdirectory(log)
# Core Library
add_subdirectory(core)
# TFlite Library
add_subdirectory(tflite)
# Tools
add_subdirectory(tools)
add_subdirectory(tests)
