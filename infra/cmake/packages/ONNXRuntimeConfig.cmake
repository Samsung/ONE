set(ONNXRUNTIME_PREFIX
    "/usr"
    CACHE PATH "The location of pre-installed ONNX Runtime library")

# TODO Download ONNXRuntime binaries

function(_ONNXRuntime_import)
  # Find the header & lib
  find_library(
    ONNXRuntime_LIB
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_PREFIX}/lib")
  find_path(
    ONNXRuntime_INCLUDE_DIR
    NAMES onnxruntime_c_api.h
    PATHS "${ONNXRUNTIME_PREFIX}/include")

  if(NOT ONNXRuntime_LIB OR NOT ONNXRuntime_INCLUDE_DIR)
    message(STATUS "Found ONNXRuntime: FALSE")

    set(ONNXRuntime_FOUND
        FALSE
        PARENT_SCOPE)
    return()
  endif(NOT ONNXRuntime_LIB OR NOT ONNXRuntime_INCLUDE_DIR)

  # Add onnxruntime target
  if(NOT TARGET onnxruntime)
    message(STATUS "Found ONNXRuntime (include: ${ONNXRuntime_INCLUDE_DIR}, library: ${ONNXRuntime_LIB})")

    add_library(onnxruntime INTERFACE)
    target_link_libraries(onnxruntime INTERFACE ${ONNXRuntime_LIB})
    target_include_directories(onnxruntime INTERFACE ${ONNXRuntime_INCLUDE_DIR})
  endif(NOT TARGET onnxruntime)

  set(ONNXRuntime_FOUND
      TRUE
      PARENT_SCOPE)
endfunction(_ONNXRuntime_import)

_ONNXRuntime_import()
