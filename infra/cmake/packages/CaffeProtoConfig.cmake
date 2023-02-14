function(_CaffeProto_import)
  nnas_find_package(CaffeSource QUIET)

  if(NOT CaffeSource_FOUND)
    set(CaffeProto_FOUND
        FALSE
        PARENT_SCOPE)
    return()
  endif(NOT CaffeSource_FOUND)

  nnas_find_package(Protobuf QUIET)

  if(NOT Protobuf_FOUND)
    set(CaffeProto_FOUND
        FALSE
        PARENT_SCOPE)
    return()
  endif(NOT Protobuf_FOUND)

  if(NOT TARGET caffeproto)
    nnas_include(ExternalProjectTools)
    add_extdirectory("${CMAKE_CURRENT_LIST_DIR}/CaffeProto" caffeproto)
  endif(NOT TARGET caffeproto)

  set(CaffeProto_FOUND
      TRUE
      PARENT_SCOPE)
endfunction(_CaffeProto_import)

_CaffeProto_import()
