nnas_find_package(Protobuf QUIET)

if(NOT Protobuf_FOUND)
  message(STATUS "Build tflchef: FAILED (missing Protobuf)")
  return()
endif(NOT Protobuf_FOUND)

if(NOT TARGET mio_tflite260)
  message(STATUS "Build tflchef: FAILED (missing mio_tflite260)")
  return()
endif(NOT TARGET mio_tflite260)

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
