# TargetRequire_Check(NAME t1 t2 t3 ...)
#
# TargetRequire_Check(NAME ...) sets "NAME" as TRUE if all the required targets are
# available, and FALSE otherwise.
function(TargetRequire_Check VARNAME)
  set(${VARNAME} TRUE PARENT_SCOPE)
  foreach(REQUIRED_TARGET IN ITEMS ${ARGN})
    if(NOT TARGET ${REQUIRED_TARGET})
      set(${VARNAME} FALSE PARENT_SCOPE)
      return()
    endif(NOT TARGET ${REQUIRED_TARGET})
  endforeach(REQUIRED_TARGET)
endfunction(TargetRequire_Check)

# TargetRequire_Assert(t1 t2 t3 ...)
#
# TargetRequire_Assert(...) stops CMake immediately if there is a target required but unavailable.
function(TargetRequire_Assert)
  unset(MISSING_TARGETS)

  foreach(REQUIRED_TARGET IN ITEMS ${ARGN})
    if(NOT TARGET ${REQUIRED_TARGET})
      list(APPEND MISSING_TARGETS ${REQUIRED_TARGET})
    endif(NOT TARGET ${REQUIRED_TARGET})
  endforeach(REQUIRED_TARGET)

  list(LENGTH MISSING_TARGETS MISSING_COUNT)

  if(NOT MISSING_COUNT EQUAL 0)
    message(FATAL_ERROR "${MISSING_TARGETS} are required, but unavailable")
  endif(NOT MISSING_COUNT EQUAL 0)
endfunction(TargetRequire_Assert)

# TargetRequire_Return(t1 t2 t3 ...)
#
# TargetRequire_Return(...) returns immediately if there is a target required but unavailable.
#
# NOTE "macro" is inevitable to make "return" inside affect the caller.
macro(TargetRequire_Return)
  foreach(REQUIRED_TARGET IN ITEMS ${ARGN})
    if(NOT TARGET ${REQUIRED_TARGET})
      return()
    endif(NOT TARGET ${REQUIRED_TARGET})
  endforeach(REQUIRED_TARGET)
endmacro(TargetRequire_Return)
