function(_GEMMLowpSource_import)
  if(NOT DOWNLOAD_GEMMLOWP)
    set(GEMMLowpSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_GEMMLOWP)

  nnfw_include(ExternalSourceTools)
  nnfw_include(OptionTools)

  # Exact version used by TensorFlow v2.19.1.
  # See tensorflow/third_party/gemmlowp/workspace.bzl.
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(TGEMMLOWP_URL ${EXTERNAL_DOWNLOAD_SERVER}/google/gemmlowp/archive/16e8662c34917be0065110bfcd9cc27d30f52fdf.zip)

  ExternalSource_Download(GEMMLOWP DIRNAME GEMMLOWP ${TGEMMLOWP_URL})

  set(GEMMLowpSource_DIR ${GEMMLOWP_SOURCE_DIR} PARENT_SCOPE)
  set(GEMMLowpSource_FOUND TRUE PARENT_SCOPE)
endfunction(_GEMMLowpSource_import)

_GEMMLowpSource_import()
