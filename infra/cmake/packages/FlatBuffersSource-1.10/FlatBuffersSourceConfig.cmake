function(_FlatBuffersSource_import)
  if(NOT DOWNLOAD_FLATBUFFERS)
    set(FlatBuffersSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_FLATBUFFERS)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # NOTE TensorFlow 1.12 downloads flatbuffers commit ID 1f5eae5d6a135ff6811724f6c57f911d1f46bb15
  #      TensorFlow 1.13.1 downloads flatbuffers commit ID 1f5eae5d6a135ff6811724f6c57f911d1f46bb15
  #      The closest release with 1f5eae5d6a135ff6811724f6c57f911d1f46bb15 is v1.10
  envoption(FLATBUFFERS_1_10_URL https://github.com/google/flatbuffers/archive/v1.10.0.tar.gz)

  ExternalSource_Download(FLATBUFFERS DIRNAME FLATBUFFERS-1.10 ${FLATBUFFERS_1_10_URL})

  set(FlatBuffersSource_DIR ${FLATBUFFERS_SOURCE_DIR} PARENT_SCOPE)
  set(FlatBuffersSource_FOUND TRUE PARENT_SCOPE)
endfunction(_FlatBuffersSource_import)

_FlatBuffersSource_import()
