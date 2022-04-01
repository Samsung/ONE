# Looking for pre-installed TRIX engine package
set(TRIX_ENGINE_PREFIX "/usr" CACHE PATH "Where to find TRIX engine header and library")

function(_TRIXEngine_import)
  # Find the header & lib
  find_library(TRIXEngine_LIB
    NAMES npu-engine
    PATHS "${TRIX_ENGINE_PREFIX}/lib"
  )

  find_path(TRIXEngine_INCLUDE_DIR
    NAMES libnpuhost.h
    PATHS "${TRIX_ENGINE_PREFIX}/include/npu-engine"
  )

  set(TRIXEngine_FOUND TRUE)

  if(NOT TRIXEngine_LIB)
    set(TRIXEngine_FOUND FALSE)
  endif(NOT TRIXEngine_LIB)

  if(NOT TRIXEngine_INCLUDE_DIR)
    set(TRIXEngine_FOUND FALSE)
  endif(NOT TRIXEngine_INCLUDE_DIR)

  if(NOT TRIXEngine_FOUND)
    message(STATUS "Failed to find TRIX Engine")
  else(NOT TRIXEngine_FOUND)

    # Add target
    if(NOT TARGET npu_engine)
      add_library(npu_engine INTERFACE)
      target_link_libraries(npu_engine INTERFACE ${TRIXEngine_LIB})
      target_include_directories(npu_engine INTERFACE ${TRIXEngine_INCLUDE_DIR})
    endif(NOT TARGET npu_engine)
  endif(NOT TRIXEngine_FOUND)

  set(TRIXEngine_FOUND ${TRIXEngine_FOUND} PARENT_SCOPE)
endfunction(_TRIXEngine_import)

_TRIXEngine_import()
