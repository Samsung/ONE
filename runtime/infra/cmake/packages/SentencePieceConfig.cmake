set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

function(_SentencePiece_import)
  if(TARGET sentencepiece::sentencepiece)
    # Already found
    return()
  endif()

  nnfw_find_package(SentencePieceSource QUIET)

  if(NOT SentencePieceSource_FOUND)
    set(SentencePiece_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT SentencePieceSource_FOUND)

  include_directories(${SentencePieceSource_DIR})
  nnfw_include(ExternalProjectTools)
  add_extdirectory(${SentencePieceSource_DIR} sentencepiece EXCLUDE_FROM_ALL)
  if(NOT TARGET sentencepiece)
    set(SentencePiece_FOUND FALSE PARENT_SCOPE)
    return()
  endif()

  set_property(TARGET sentencepiece PROPERTY POSITION_INDEPENDENT_CODE ON)
  add_library(sentencepiece::sentencepiece ALIAS sentencepiece)

  # Install SentencePiece library to the same directory as ggma
  install(TARGETS sentencepiece
          LIBRARY DESTINATION ${GGMA_INSTALL_LIBDIR})

  set(SentencePiece_FOUND TRUE PARENT_SCOPE)
  set(SentencePiece_LIBRARIES sentencepiece PARENT_SCOPE)
  set(SentencePiece_INCLUDE_DIRS ${SentencePieceSource_DIR}/src PARENT_SCOPE)
endfunction(_SentencePiece_import)

_SentencePiece_import()
