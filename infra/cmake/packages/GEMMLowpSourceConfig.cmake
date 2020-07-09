function(_GEMMLowpSource_import)
  if(NOT DOWNLOAD_GEMMLOWP)
    set(GEMMLowpSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_GEMMLOWP)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # NOTE TensorFlow 1.13.1 uses the following URL
  envoption(GEMMLOWP_URL https://github.com/google/gemmlowp/archive/38ebac7b059e84692f53e5938f97a9943c120d98.tar.gz)

  ExternalSource_Download(GEMMLOWP ${GEMMLOWP_URL})

  set(GEMMLowpSource_DIR ${GEMMLOWP_SOURCE_DIR} PARENT_SCOPE)
  set(GEMMLowpSource_FOUND TRUE PARENT_SCOPE)
endfunction(_GEMMLowpSource_import)

_GEMMLowpSource_import()
