function(_FlatBuffersSource_import)
  if(NOT DOWNLOAD_FLATBUFFERS)
    set(FlatBuffersSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_FLATBUFFERS)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  set(FLATBUFFERS_URL https://github.com/google/flatbuffers/archive/v1.11.0.tar.gz)

  ExternalSource_Download(FLATBUFFERS DIRNAME FLATBUFFERS-1.11 ${FLATBUFFERS_URL})

  set(FlatBuffersSource_DIR ${FLATBUFFERS_SOURCE_DIR} PARENT_SCOPE)
  set(FlatBuffersSource_FOUND TRUE PARENT_SCOPE)
endfunction(_FlatBuffersSource_import)

_FlatBuffersSource_import()
