# Looking for pre-installed NPU engine package
set(NPU_ENGINE_PREFIX "/usr" CACHE PATH "Where to find NPU engine header and library")

function(_NPUEngine_import)
  # Find the header & lib
  find_library(NPUEngine_LIB
    NAMES npu-engine
    PATHS "${NPU_ENGINE_PREFIX}/lib"
  )

  find_path(NPUEngine_INCLUDE_DIR
    NAMES libnpuhost.h
    PATHS "${NPU_ENGINE_PREFIX}/include/npu-engine"
  )

  set(NPUEngine_FOUND TRUE)

  if(NOT NPUEngine_LIB)
    set(NPUEngine_FOUND FALSE)
  endif(NOT NPUEngine_LIB)

  if(NOT NPUEngine_INCLUDE_DIR)
    set(NPUEngine_FOUND FALSE)
  endif(NOT NPUEngine_INCLUDE_DIR)

  if(NOT NPUEngine_FOUND)
    message(STATUS "Failed to find NPU Engine")
  else(NOT NPUEngine_FOUND)

    # Add target
    if(NOT TARGET npu_engine)
      add_library(npu_engine INTERFACE)
      target_link_libraries(npu_engine INTERFACE ${NPUEngine_LIB})
      target_include_directories(npu_engine INTERFACE ${NPUEngine_INCLUDE_DIR})
    endif(NOT TARGET npu_engine)
  endif(NOT NPUEngine_FOUND)

  set(NPUEngine_FOUND ${NPUEngine_FOUND} PARENT_SCOPE)
endfunction(_NPUEngine_import)

_NPUEngine_import()
