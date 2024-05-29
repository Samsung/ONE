function(_MLDtypesSource_import)
  if(NOT DOWNLOAD_MLDTYPES)
    set(MLDtypesSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_MLDTYPES)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(MLDTYPES_URL ${EXTERNAL_DOWNLOAD_SERVER}/jax-ml/ml_dtypes/archive/2ca30a2b3c0744625ae3d6988f5596740080bbd0/ml_dtypes-2ca30a2b3c0744625ae3d6988f5596740080bbd0.tar.gz)

  ExternalSource_Download(MLDTYPES ${MLDTYPES_URL})

  set(MLDtypesSource_DIR ${MLDTYPES_SOURCE_DIR} PARENT_SCOPE)
  set(MLDtypesSource_FOUND TRUE PARENT_SCOPE)
endfunction(_MLDtypesSource_import)

_MLDtypesSource_import()
