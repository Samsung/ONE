# NOTE This line prevents multiple definitions of tensorflow-lite target
if(TARGET tensorflow-lite)
  set(TensorFlowLite_FOUND TRUE)
  return()
endif(TARGET tensorflow-lite)

if(BUILD_TENSORFLOW_LITE)
  macro(return_unless VAR)
    if(NOT ${VAR})
      message("TFLite 2.18.1: ${VAR} NOT TRUE")
      set(TensorFlowLite_FOUND FALSE)
      return()
    endif(NOT ${VAR})
  endmacro(return_unless)

  message(STATUS "Building TFLite 2.18.1...")

  nnfw_include(ExternalSourceTools)
  nnfw_include(OptionTools)

  nnfw_find_package(TensorFlowSource QUIET)
  return_unless(TensorFlowSource_FOUND)

  # Below urls come from https://github.com/tensorflow/tensorflow/blob/v2.18.1/tensorflow/workspace2.bzl
  nnfw_find_package(Abseil QUIET)
  return_unless(Abseil_FOUND)
  nnfw_find_package(Eigen QUIET)
  return_unless(Eigen_FOUND)
  nnfw_find_package(Farmhash QUIET)
  return_unless(Farmhash_FOUND)
  nnfw_find_package(FlatBuffers 23.5.26 QUIET)
  return_unless(FlatBuffers_FOUND)
  nnfw_find_package(GEMMLowpSource QUIET)
  return_unless(GEMMLowpSource_FOUND)
  nnfw_find_package(OouraFFT QUIET)
  return_unless(OouraFFT_FOUND)
  nnfw_find_package(Ruy QUIET)
  return_unless(Ruy_FOUND)
  nnfw_find_package(MLDtypesSource QUIET)
  return_unless(MLDtypesSource_FOUND)

  # TensorFlow Lite requires FP16 library's header only
  nnfw_find_package(Fp16Source QUIET)
  return_unless(Fp16Source_FOUND)

  # PThreadpool
  nnfw_find_package(Pthreadpool QUIET)
  return_unless(Pthreadpool_FOUND)

  # TensorFlow Lite requires Pybind11 library's header only
  # But Pybind11 requires python3-dev package
  # TODO Enable below by installing package on build system
  #nnfw_find_package(Pybind11Source QUIET)
  #return_unless(Pybind11Source_FOUND)

  # Optional packages
  nnfw_find_package(NEON2SSESource QUIET)

  nnfw_include(ExternalProjectTools)
  add_extdirectory("${CMAKE_CURRENT_LIST_DIR}/TensorFlowLite" tflite-2.18.1)

  set(TensorFlowLite_FOUND TRUE)
  return()
endif()

# Use pre-built TensorFlow Lite
find_path(TFLITE_INCLUDE_DIR NAMES  tensorflow/lite/c/c_api.h)
find_library(TFLITE_LIB NAMES       tensorflow2-lite)

if(NOT TFLITE_INCLUDE_DIR)
  # Tizen install TensorFlow Lite 2.x headers in /usr/include/tensorflow2
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
add_library(tensorflow-lite INTERFACE)
target_include_directories(tensorflow-lite SYSTEM INTERFACE ${TFLITE_INCLUDE_DIR})
target_link_libraries(tensorflow-lite INTERFACE ${TFLITE_LIB})
find_package(Flatbuffers)
if(Flatbuffers_FOUND)
  target_link_libraries(tensorflow-lite INTERFACE flatbuffers::flatbuffers)
endif(Flatbuffers_FOUND)

target_link_libraries(tensorflow-lite INTERFACE Threads::Threads ${CMAKE_DL_LIBS})

set(TensorFlowLite_FOUND TRUE)
