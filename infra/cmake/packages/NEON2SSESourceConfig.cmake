function(_NEON2SSESource_import)
  if(NOT DOWNLOAD_NEON2SSE)
    set(NEON2SSESource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_NEON2SSE)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # NOTE TensorFlow 1.7 downloads NEON2SSE from the following URL
  envoption(NEON2SSE_URL https://github.com/intel/ARM_NEON_2_x86_SSE/archive/0f77d9d182265259b135dad949230ecbf1a2633d.tar.gz)

  ExternalSource_Download(NEON2SSE ${NEON2SSE_URL})

  set(NEON2SSESource_DIR ${NEON2SSE_SOURCE_DIR} PARENT_SCOPE)
  set(NEON2SSESource_FOUND TRUE PARENT_SCOPE)
endfunction(_NEON2SSESource_import)

_NEON2SSESource_import()
