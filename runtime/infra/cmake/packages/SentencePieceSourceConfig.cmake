function(_SentencePieceSource_import)
  if(NOT DOWNLOAD_SENTENCEPIECE)
    set(SentencePieceSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_SENTENCEPIECE)

  nnfw_include(ExternalSourceTools)
  nnfw_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(SENTENCEPIECE_URL ${EXTERNAL_DOWNLOAD_SERVER}/google/sentencepiece/archive/v0.1.90.tar.gz)

  ExternalSource_Download(SENTENCEPIECE
    DIRNAME SENTENCEPIECE
    URL ${SENTENCEPIECE_URL}
  )

  set(SentencePieceSource_DIR ${SENTENCEPIECE_SOURCE_DIR} PARENT_SCOPE)
  set(SentencePieceSource_FOUND TRUE PARENT_SCOPE)
endfunction(_SentencePieceSource_import)

_SentencePieceSource_import()
