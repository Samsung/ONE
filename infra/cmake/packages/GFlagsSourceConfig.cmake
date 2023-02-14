function(_GFlagsSource_import)
  if(NOT DOWNLOAD_GFLAGS)
    set(GFlagsSource_FOUND
        False
        PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_GFLAGS)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(GFLAGS_URL ${EXTERNAL_DOWNLOAD_SERVER}/gflags/gflags/archive/v2.2.1.tar.gz)

  ExternalSource_Download(GFLAGS ${GFLAGS_URL})

  set(GFLAGS_SOURCE_DIR
      ${GFLAGS_SOURCE_DIR}
      PARENT_SCOPE)
  set(GFlagsSource_FOUND
      True
      PARENT_SCOPE)
endfunction(_GFlagsSource_import)

_GFlagsSource_import()
