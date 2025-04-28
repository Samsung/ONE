function(_PsimdSource_import)
  if(NOT ${DOWNLOAD_PSIMD})
    set(PsimdSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${DOWNLOAD_PSIMD})

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  # psimd commit in xnnpack 8b283aa30a31
  envoption(PSIMD_URL ${EXTERNAL_DOWNLOAD_SERVER}/Maratyszcza/psimd/archive/072586a71b55b7f8c584153d223e95687148a900.tar.gz)
  ExternalSource_Download(PSIMD
    DIRNAME PSIMD
    URL ${PSIMD_URL})

  set(PsimdSource_DIR ${PSIMD_SOURCE_DIR} PARENT_SCOPE)
  set(PsimdSource_FOUND TRUE PARENT_SCOPE)
endfunction(_PsimdSource_import)

_PsimdSource_import()
