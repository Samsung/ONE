function(_Psimd_Build)
  nnas_find_package(PsimdSource QUIET)

  # NOTE This line prevents multiple definitions of target
  if(TARGET psimd)
    set(PsimdSource_DIR ${PsimdSource_DIR} PARENT_SCOPE)
    set(Psimd_FOUND TRUE PARENT_SCOPE)
    return()
  endif(TARGET psimd)

  if(NOT PsimdSource_FOUND)
    message(STATUS "PSIMD: Source not found")
    set(Psimd_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT PsimdSource_FOUND)

  add_extdirectory("${PsimdSource_DIR}" PSIMD EXCLUDE_FROM_ALL)
  set(PsimdSource_DIR ${PsimdSource_DIR} PARENT_SCOPE)
  set(Psimd_FOUND TRUE PARENT_SCOPE)
endfunction(_Psimd_Build)

if(BUILD_PSIMD)
  _Psimd_Build()
else()
  set(Psimd_FOUND FALSE)
endif()
