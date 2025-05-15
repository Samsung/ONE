function(_NEON2SSESource_import)
  if(NOT DOWNLOAD_NEON2SSE)
    set(NEON2SSESource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_NEON2SSE)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # NOTE TensorFlow 2.16.1 downloads NEON2SSE from the following URL
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(NEON2SSE_URL ${EXTERNAL_DOWNLOAD_SERVER}/intel/ARM_NEON_2_x86_SSE/archive/a15b489e1222b2087007546b4912e21293ea86ff.tar.gz)

  ExternalSource_Download(NEON2SSE ${NEON2SSE_URL})

  set(NEON2SSESource_DIR ${NEON2SSE_SOURCE_DIR} PARENT_SCOPE)
  set(NEON2SSESource_FOUND TRUE PARENT_SCOPE)
endfunction(_NEON2SSESource_import)

_NEON2SSESource_import()
