nnas_find_package(Protobuf QUIET)

if(NOT Protobuf_FOUND)
  message(STATUS "Build tflchef: FAILED (missing Protobuf)")
  return()
endif(NOT Protobuf_FOUND)

if(NOT TARGET mio_tflite2121)
  message(STATUS "Build tflchef: FAILED (missing mio_tflite2121)")
  return()
endif(NOT TARGET mio_tflite2121)

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

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

add_subdirectory(tests)
