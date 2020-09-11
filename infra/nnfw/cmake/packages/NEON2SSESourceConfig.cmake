function(_NEON2SSESource_import)
  if(NOT ${DOWNLOAD_NEON2SSE})
    set(NEON2SSESource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${DOWNLOAD_NEON2SSE})

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # NOTE TensorFlow 1.12 downloads NEON2SSE from the following URL
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  set(NEON2SSE_URL ${EXTERNAL_DOWNLOAD_SERVER}/intel/ARM_NEON_2_x86_SSE/archive/0f77d9d182265259b135dad949230ecbf1a2633d.tar.gz)
  ExternalSource_Download("neon_2_sse" ${NEON2SSE_URL})

  set(NEON2SSESource_DIR ${neon_2_sse_SOURCE_DIR} PARENT_SCOPE)
  set(NEON2SSESource_FOUND TRUE PARENT_SCOPE)
endfunction(_NEON2SSESource_import)

_NEON2SSESource_import()
