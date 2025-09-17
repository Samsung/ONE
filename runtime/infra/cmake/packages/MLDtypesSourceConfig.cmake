function(_MLDtypesSource_import)
  if(NOT DOWNLOAD_MLDTYPES)
    set(MLDtypesSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_MLDTYPES)

  nnfw_include(ExternalSourceTools)
  nnfw_include(OptionTools)

  # ml_dtypes in tensorflow v2.18.1: refer third_party/xla/third_party/tsl/third_party/py/ml_dtypes/workspace.bzl
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(MLDTYPES_URL ${EXTERNAL_DOWNLOAD_SERVER}/jax-ml/ml_dtypes/archive/6f02f77c4fa624d8b467c36d1d959a9b49b07900/ml_dtypes-6f02f77c4fa624d8b467c36d1d959a9b49b07900.tar.gz)

  ExternalSource_Download(MLDTYPES ${MLDTYPES_URL})

  set(MLDtypesSource_DIR ${MLDTYPES_SOURCE_DIR} PARENT_SCOPE)
  set(MLDtypesSource_FOUND TRUE PARENT_SCOPE)
endfunction(_MLDtypesSource_import)

_MLDtypesSource_import()
