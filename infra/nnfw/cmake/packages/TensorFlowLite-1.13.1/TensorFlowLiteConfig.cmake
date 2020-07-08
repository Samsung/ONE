# NOTE This line prevents multiple definitions of tensorflow-lite target
if(TARGET tensorflow-lite)
  set(TensorFlowLite_FOUND TRUE)
  return()
endif(TARGET tensorflow-lite)

if(BUILD_TENSORFLOW_LITE)
  macro(return_unless VAR)
    if(NOT ${VAR})
      set(TensorFlowLite_FOUND PARENT_SCOPE)
      return()
    endif(NOT ${VAR})
  endmacro(return_unless)

  # Required packages
  nnas_find_package(AbseilSource QUIET)
  return_unless(AbseilSource_FOUND)
  nnfw_find_package(Eigen QUIET)
  return_unless(Eigen_FOUND)
  nnfw_find_package(FarmhashSource QUIET)
  return_unless(FarmhashSource_FOUND)
  nnfw_find_package(FlatBuffersSource QUIET)
  return_unless(FlatBuffersSource_FOUND)
  nnfw_find_package(GEMMLowpSource QUIET)
  return_unless(GEMMLowpSource_FOUND)
  nnas_find_package(TensorFlowSource EXACT 1.13.1 QUIET)
  return_unless(TensorFlowSource_FOUND)

  # Optional packages
  nnfw_find_package(NEON2SSESource QUIET)

  nnas_include(ExternalProjectTools)
  add_extdirectory("${CMAKE_CURRENT_LIST_DIR}/TensorFlowLite" tflite)

  set(TensorFlowLite_FOUND TRUE)
  return()
endif(BUILD_TENSORFLOW_LITE)

# Use pre-built TensorFlow Lite
find_path(TFLITE_INCLUDE_DIR NAMES  tensorflow/lite/interpreter.h)
find_library(TFLITE_LIB NAMES       tensorflow-lite)

if(NOT TFLITE_INCLUDE_DIR)
  set(TensorFlowLite_FOUND FALSE)
  return()
endif(NOT TFLITE_INCLUDE_DIR)

if(NOT TFLITE_LIB)
  set(TensorFlowLite_FOUND FALSE)
  return()
endif(NOT TFLITE_LIB)

message(STATUS "Found TensorFlow Lite: TRUE (include: ${TFLITE_INCLUDE_DIR}, lib: ${TFLITE_LIB}")

# TODO Use IMPORTED target
add_library(tensorflow-lite INTERFACE)
target_include_directories(tensorflow-lite SYSTEM INTERFACE ${TFLITE_INCLUDE_DIR})
target_link_libraries(tensorflow-lite INTERFACE ${TFLITE_LIB})
find_package(Flatbuffers)
if(Flatbuffers_FOUND)
  target_link_libraries(tensorflow-lite INTERFACE flatbuffers::flatbuffers)
endif(Flatbuffers_FOUND)

# Prefer -pthread to -lpthread
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads QUIET)

if(Threads_FOUND)
  target_link_libraries(tensorflow-lite INTERFACE ${CMAKE_THREAD_LIBS_INIT})
endif(Threads_FOUND)

set(TensorFlowLite_FOUND TRUE)
