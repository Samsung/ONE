function(_ProtobufSource_import)
  if(NOT DOWNLOAD_PROTOBUF)
    set(ProtobufSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_PROTOBUF)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(PROTOBUF_URL https://github.com/protocolbuffers/protobuf/archive/v3.5.2.tar.gz)

  ExternalSource_Download(PROTOBUF ${PROTOBUF_URL}
                          PATCH ${CMAKE_CURRENT_LIST_DIR}/ProtobufSource.patch)

  set(ProtobufSource_DIR ${PROTOBUF_SOURCE_DIR} PARENT_SCOPE)
  set(ProtobufSource_FOUND TRUE PARENT_SCOPE)
endfunction(_ProtobufSource_import)

_ProtobufSource_import()
