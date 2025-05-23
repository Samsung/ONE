# Read a file and create a list variable
#
# HOW TO USE
#
#  ListFile_Read("A.txt" A_LIST)
#
function(ListFile_Read FILENAME VARNAME)
  file(READ ${FILENAME} content)
  # Reference: http://public.kitware.com/pipermail/cmake/2007-May/014236.html
  STRING(REGEX REPLACE "\n" ";" content "${content}")
  set(${VARNAME} ${content} PARENT_SCOPE)
endfunction(ListFile_Read)
