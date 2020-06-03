function(_FlatBuffersSource_import)
  if(NOT ${DOWNLOAD_FLATBUFFERS})
    set(FlatBuffersSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${DOWNLOAD_FLATBUFFERS})

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # NOTE TensorFlow 1.12 downloads FlatBuffers from the following URL
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  set(FLATBUFFERS_URL ${EXTERNAL_DOWNLOAD_SERVER}/google/flatbuffers/archive/1f5eae5d6a135ff6811724f6c57f911d1f46bb15.tar.gz)
  ExternalSource_Download("flatbuffers" ${FLATBUFFERS_URL})

  set(FlatBuffersSource_DIR ${flatbuffers_SOURCE_DIR} PARENT_SCOPE)
  set(FlatBuffersSource_FOUND ${flatbuffers_SOURCE_GET} PARENT_SCOPE)
endfunction(_FlatBuffersSource_import)

_FlatBuffersSource_import()
