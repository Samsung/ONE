function(_FlatBuffersSource_import)
  if(NOT DOWNLOAD_FLATBUFFERS)
    set(FlatBuffersSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_FLATBUFFERS)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(FLATBUFFERS_2_0_URL ${EXTERNAL_DOWNLOAD_SERVER}/google/flatbuffers/archive/v2.0.0.tar.gz)
  ExternalSource_Download(FLATBUFFERS
    DIRNAME FLATBUFFERS-2.0
    CHECKSUM MD5=a27992324c3cbf86dd888268a23d17bd
    URL ${FLATBUFFERS_2_0_URL}
  )

  set(FlatBuffersSource_DIR ${FLATBUFFERS_SOURCE_DIR} PARENT_SCOPE)
  set(FlatBuffersSource_FOUND TRUE PARENT_SCOPE)
endfunction(_FlatBuffersSource_import)

_FlatBuffersSource_import()
