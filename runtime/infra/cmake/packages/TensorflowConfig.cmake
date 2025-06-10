function(_Tensorflow_Import)
  if(NOT DEFINED TENSORFLOW_DIR)
    set(TENSORFLOW_DIR ${NNFW_EXTERNALS_DIR}/tensorflow)
  endif(NOT DEFINED TENSORFLOW_DIR)

  if(NOT DEFINED NSYNC_ARCH)
    set(NSYNC_ARCH "default")
  endif(NOT DEFINED NSYNC_ARCH)

  set(TENSROFLOW_MAKEFILE_DIR "${TENSORFLOW_DIR}/tensorflow/makefile")
  set(TENSORFLOW_GEN_DIR "${TENSROFLOW_MAKEFILE_DIR}/gen")
  set(TENSORFLOW_DOWNLOADS_DIR "${TENSROFLOW_MAKEFILE_DIR}/downloads")

  if(NOT EXISTS "${TENSORFLOW_GEN_DIR}/lib/libtensorflow-core.a")
    set(Tensorflow_FOUND FALSE PARENT_SCOPE)
    return()
  endif()

  if(NOT EXISTS "${TENSORFLOW_DOWNLOADS_DIR}/nsync/builds/${NSYNC_ARCH}.linux.c++11/libnsync.a")
    set(Tensorflow_FOUND FALSE PARENT_SCOPE)
    return()
  endif()

  if(NOT TARGET tensorflow-core)
    add_library(tensorflow-core INTERFACE)

    target_include_directories(tensorflow-core INTERFACE "${TENSORFLOW_DIR}")
    target_include_directories(tensorflow-core INTERFACE "${TENSORFLOW_GEN_DIR}/proto")
    target_include_directories(tensorflow-core INTERFACE "${TENSORFLOW_GEN_DIR}/protobuf/include")
    target_include_directories(tensorflow-core INTERFACE "${TENSORFLOW_DOWNLOADS_DIR}/eigen")
    target_include_directories(tensorflow-core INTERFACE "${TENSORFLOW_DOWNLOADS_DIR}/nsync/public")

    target_link_libraries(tensorflow-core INTERFACE -Wl,--whole-archive "${TENSORFLOW_GEN_DIR}/lib/libtensorflow-core.a" -Wl,--no-whole-archive)
    target_link_libraries(tensorflow-core INTERFACE "${TENSORFLOW_GEN_DIR}/protobuf/lib/libprotobuf.a")
    target_link_libraries(tensorflow-core INTERFACE "${TENSORFLOW_DOWNLOADS_DIR}/nsync/builds/${NSYNC_ARCH}.linux.c++11/libnsync.a")
    target_link_libraries(tensorflow-core INTERFACE ${LIB_PTHREAD} dl)

    message(STATUS "Found Tensorflow (lib: ${TENSORFLOW_GEN_DIR}/lib/libtensorflow-core.a")
  endif()

    set(Tensorflow_FOUND TRUE PARENT_SCOPE)
endfunction(_Tensorflow_Import)

_Tensorflow_Import()
