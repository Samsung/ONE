function(_FlatBuffersSource_import)
  if(NOT DOWNLOAD_FLATBUFFERS)
    set(FlatBuffersSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_FLATBUFFERS)

  nnfw_include(ExternalSourceTools)
  nnfw_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(FLATBUFFERS_URL ${EXTERNAL_DOWNLOAD_SERVER}/google/flatbuffers/archive/v25.2.10.tar.gz)
  ExternalSource_Download(FLATBUFFERS
    DIRNAME FLATBUFFERS
    URL ${FLATBUFFERS_URL}
  )

  set(FlatBuffersSource_DIR ${FLATBUFFERS_SOURCE_DIR} PARENT_SCOPE)
  set(FlatBuffersSource_FOUND TRUE PARENT_SCOPE)
endfunction(_FlatBuffersSource_import)

_FlatBuffersSource_import()
