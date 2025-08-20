function(_ProtobufSource_import)
  if(NOT DOWNLOAD_PROTOBUF)
    set(ProtobufSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_PROTOBUF)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(PROTOBUF_URL ${EXTERNAL_DOWNLOAD_SERVER}/protocolbuffers/protobuf/archive/v3.21.12.tar.gz)

  ExternalSource_Download(PROTOBUF ${PROTOBUF_URL})

  set(ProtobufSource_DIR ${PROTOBUF_SOURCE_DIR} PARENT_SCOPE)
  set(ProtobufSource_FOUND TRUE PARENT_SCOPE)
endfunction(_ProtobufSource_import)

_ProtobufSource_import()
