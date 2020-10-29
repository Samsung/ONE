set(TENSORFLOW_PREFIX "/usr" CACHE PATH "The location of pre-installed TensorFlow 1.13 library")
set(TENSORFLOW_VERSION_REQUIRED "1.13")

# TODO Build TensorFlow from the (downloaded) source

function(_TensorFlow_import)
  # Find the header & lib
  find_library(TensorFlow_LIB NAMES tensorflow PATHS "${TENSORFLOW_PREFIX}/lib")
  find_path(TensorFlow_INCLUDE_DIR NAMES tensorflow/c/c_api.h PATHS "${TENSORFLOW_PREFIX}/include")

  if(NOT TensorFlow_LIB OR NOT TensorFlow_INCLUDE_DIR)
    message(STATUS "Found TensorFlow: FALSE")

    set(TensorFlow_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT TensorFlow_LIB OR NOT TensorFlow_INCLUDE_DIR)

  # Check TensorFlow version
  try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/TensorFlowVersionChecker.c
    COMPILE_DEFINITIONS -I${TensorFlow_INCLUDE_DIR}
    LINK_LIBRARIES ${TensorFlow_LIB}
    ARGS ${TENSORFLOW_VERSION_REQUIRED})

  if(NOT COMPILE_RESULT_VAR)
    message(STATUS "Failed to build TensorFlowVersionChecker. Your libtensorflow may be built on different version of Ubuntu.")
    message(STATUS "Found TensorFlow: FALSE")
    set(TensorFlow_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT COMPILE_RESULT_VAR)

  if(NOT RUN_RESULT_VAR EQUAL 0)
    message(STATUS "you need tensorflow version ${TENSORFLOW_VERSION_REQUIRED}")
    message(STATUS "Found TensorFlow: FALSE")
    set(TensorFlow_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT RUN_RESULT_VAR EQUAL 0)

  # Add tensorflow target (if necessary)
  if(NOT TARGET tensorflow-1.13)
    message(STATUS "Found TensorFlow (include: ${TensorFlow_INCLUDE_DIR}, library: ${TensorFlow_LIB})")

    # NOTE IMPORTED target may be more appropriate for this case
    add_library(tensorflow-1.13 INTERFACE)
    target_link_libraries(tensorflow-1.13 INTERFACE ${TensorFlow_LIB})
    target_include_directories(tensorflow-1.13 INTERFACE ${TensorFlow_INCLUDE_DIR})
  endif(NOT TARGET tensorflow-1.13)

  set(TensorFlow_FOUND TRUE PARENT_SCOPE)
endfunction(_TensorFlow_import)

_TensorFlow_import()
