function(_NEON2SSESource_import)
  # TODO Remove this workaround once target preset is ready
  if(NOT (TARGET_ARCH_BASE STREQUAL "x86_64"))
    set(NEON2SSESource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT (TARGET_ARCH_BASE STREQUAL "x86_64"))

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # NOTE TensorFlow 1.12 downloads NEON2SSE from the following URL
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  set(NEON2SSE_URL ${EXTERNAL_DOWNLOAD_SERVER}/intel/ARM_NEON_2_x86_SSE/archive/0f77d9d182265259b135dad949230ecbf1a2633d.tar.gz)
  ExternalSource_Get("neon_2_sse" ${DOWNLOAD_NEON2SSE} ${NEON2SSE_URL})

  set(NEON2SSESource_DIR ${neon_2_sse_SOURCE_DIR} PARENT_SCOPE)
  set(NEON2SSESource_FOUND ${neon_2_sse_SOURCE_GET} PARENT_SCOPE)
endfunction(_NEON2SSESource_import)

_NEON2SSESource_import()
