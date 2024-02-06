function(_FlatBuffersSource_import)
  if(NOT DOWNLOAD_FLATBUFFERS)
    set(FlatBuffersSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_FLATBUFFERS)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(FLATBUFFERS_23_5_26_URL ${EXTERNAL_DOWNLOAD_SERVER}/google/flatbuffers/archive/v23.5.26.tar.gz)
  ExternalSource_Download(FLATBUFFERS
    DIRNAME FLATBUFFERS-23.5.26
    CHECKSUM MD5=2ef00eaaa86ab5e9ad5eafe09c2e7b60
    URL ${FLATBUFFERS_23_5_26_URL}
  )

  set(FlatBuffersSource_DIR ${FLATBUFFERS_SOURCE_DIR} PARENT_SCOPE)
  set(FlatBuffersSource_FOUND TRUE PARENT_SCOPE)
endfunction(_FlatBuffersSource_import)

_FlatBuffersSource_import()
