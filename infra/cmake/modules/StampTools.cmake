# Stamp_Check(VARNAME PATH CONTENT)
#  Stamp_Check sets VARNAME as TRUE if a file exists at "PATH", and its content is same as "CONTENT"
#  Stamp_Check sets VARNAME as FALSE otherwise
function(Stamp_Check VARNAME PATH EXPECTED_CONTENT)
  if(NOT EXISTS "${PATH}")
    set(${VARNAME} FALSE PARENT_SCOPE)
    return()
  endif(NOT EXISTS "${PATH}")

  file(READ ${PATH} OBTAINED_CONTENT)

  if(NOT EXPECTED_CONTENT STREQUAL OBTAINED_CONTENT)
    set(${VARNAME} FALSE PARENT_SCOPE)
    return()
  endif(NOT EXPECTED_CONTENT STREQUAL OBTAINED_CONTENT)

  set(${VARNAME} TRUE PARENT_SCOPE)
endfunction(Stamp_Check)
