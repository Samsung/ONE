function(_FlatBuffersSource_import)
  if(NOT DOWNLOAD_FLATBUFFERS)
    set(FlatBuffersSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_FLATBUFFERS)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(FLATBUFFERS_1_12_URL https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz)
  ExternalSource_Download(FLATBUFFERS
    DIRNAME FLATBUFFERS-1.12
    CHECKSUM MD5=c62ffefb3d4548b127cca14ce047f16c
    URL ${FLATBUFFERS_1_12_URL}
  )

  set(FlatBuffersSource_DIR ${FLATBUFFERS_SOURCE_DIR} PARENT_SCOPE)
  set(FlatBuffersSource_FOUND TRUE PARENT_SCOPE)
endfunction(_FlatBuffersSource_import)

_FlatBuffersSource_import()
