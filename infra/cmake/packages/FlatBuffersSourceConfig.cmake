function(_FlatBuffersSource_import)
  if(NOT DOWNLOAD_FLATBUFFERS)
    set(FlatBuffersSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_FLATBUFFERS)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # Each TensorFlow needs a specific version of Flatbuffers
  # - TensorFlow 1.7 downloads it from https://github.com/google/flatbuffers/archive/971a68110e4.tar.gz
  # - TensorFlow 1.12 downloads it from https://github.com/google/flatbuffers/archive/1f5eae5d6a1.tar.gz
  #
  # Let's use 1.10 released in 2018.10 (compatible with 1f5eae5d6a1).
  #
  # TODO Manage multiple versions
  envoption(FLATBUFFERS_URL https://github.com/google/flatbuffers/archive/v1.10.0.tar.gz)
  ExternalSource_Download(FLATBUFFERS
    DIRNAME FLATBUFFERS
    CHECKSUM MD5=f7d19a3f021d93422b0bc287d7148cd2
    URL ${FLATBUFFERS_URL}
  )

  set(FlatBuffersSource_DIR ${FLATBUFFERS_SOURCE_DIR} PARENT_SCOPE)
  set(FlatBuffersSource_FOUND TRUE PARENT_SCOPE)
endfunction(_FlatBuffersSource_import)

_FlatBuffersSource_import()
