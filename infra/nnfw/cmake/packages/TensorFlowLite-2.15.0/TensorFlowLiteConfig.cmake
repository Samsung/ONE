# NOTE This line prevents multiple definitions of tensorflow-lite target
if(TARGET tensorflow-lite-2.15.0)
  set(TensorFlowLite_FOUND TRUE)
  return()
endif(TARGET tensorflow-lite-2.15.0)

if(BUILD_TENSORFLOW_LITE)
  macro(return_unless VAR)
  if(NOT ${VAR})
    message("TFLite 2.15: ${VAR} NOT TRUE")
    set(TensorFlowLite_FOUND FALSE)
    return()
  endif(NOT ${VAR})
  endmacro(return_unless)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  nnas_find_package(TensorFlowSource EXACT 2.15.0 QUIET)
  return_unless(TensorFlowSource_FOUND)

  # Below urls come from https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/workspace2.bzl
  nnas_find_package(Abseil QUIET)
  return_unless(Abseil_FOUND)
  nnfw_find_package(Eigen QUIET)
  return_unless(Eigen_FOUND)
  nnas_find_package(Farmhash QUIET)
  return_unless(Farmhash_FOUND)
  nnfw_find_package(FlatBuffers EXACT 23.5.26 QUIET)
  return_unless(FlatBuffers_FOUND)
  nnas_find_package(TensorFlowGEMMLowpSource EXACT 2.15.0 QUIET)
  return_unless(TensorFlowGEMMLowpSource_FOUND)
  nnfw_find_package(OouraFFT QUIET)
  return_unless(OouraFFT_FOUND)
  nnfw_find_package(Ruy QUIET)
  return_unless(Ruy_FOUND)
  nnas_find_package(MLDtypesSource QUIET)
  return_unless(MLDtypesSource_FOUND)

  # TensorFlow Lite requires FP16 library's header only
  nnas_find_package(Fp16Source QUIET)
  return_unless(Fp16Source_FOUND)

  # PThreadpool
  nnfw_find_package(Pthreadpool QUIET)
  return_unless(Pthreadpool_FOUND)

  # TensorFlow Lite requires Pybind11 library's header only
  # But Pybind11 requires python3-dev package
  # TODO Enable below by installing package on build system
  #nnas_find_package(Pybind11Source QUIET)
  #return_unless(Pybind11Source_FOUND)

  # Optional packages
  nnas_find_package(NEON2SSESource QUIET)

  nnas_include(ExternalProjectTools)
  add_extdirectory("${CMAKE_CURRENT_LIST_DIR}/TensorFlowLite" tflite-2.15.0)

  set(TensorFlowLite_FOUND TRUE)
  return()
endif()

# Use pre-built TensorFlow Lite
find_path(TFLITE_INCLUDE_DIR NAMES  tensorflow/lite/c/c_api.h)
find_library(TFLITE_LIB NAMES       tensorflow2-lite)

if(NOT TFLITE_INCLUDE_DIR)
  # Tizen install TensorFlow Lite 2.8 headers in /usr/include/tensorflow2
  find_path(TFLITE_INCLUDE_DIR NAMES tensorflow/lite/c/c_api.h PATHS "/usr/include/tensorflow2")
  if(NOT TFLITE_INCLUDE_DIR)
    set(TensorFlowLite_FOUND FALSE)
    return()
  endif(NOT TFLITE_INCLUDE_DIR)
endif(NOT TFLITE_INCLUDE_DIR)

if(NOT TFLITE_LIB)
  set(TensorFlowLite_FOUND FALSE)
  return()
endif(NOT TFLITE_LIB)

message(STATUS "Found TensorFlow Lite: TRUE (include: ${TFLITE_INCLUDE_DIR}, lib: ${TFLITE_LIB}")

# TODO Use IMPORTED target
add_library(tensorflow-lite-2.15.0 INTERFACE)
target_include_directories(tensorflow-lite-2.15.0 SYSTEM INTERFACE ${TFLITE_INCLUDE_DIR})
target_link_libraries(tensorflow-lite-2.15.0 INTERFACE ${TFLITE_LIB})
find_package(Flatbuffers)
if(Flatbuffers_FOUND)
  target_link_libraries(tensorflow-lite-2.15.0 INTERFACE flatbuffers::flatbuffers)
endif(Flatbuffers_FOUND)

# Prefer -pthread to -lpthread
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads QUIET)

if(Threads_FOUND)
  target_link_libraries(tensorflow-lite-2.15.0 INTERFACE ${CMAKE_THREAD_LIBS_INIT})
endif(Threads_FOUND)

set(TensorFlowLite_FOUND TRUE)
