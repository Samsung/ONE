# AssertTarget(t) stops the build if target "t" does not exist.
function(AssertTarget TGT)
  if(TARGET ${TGT})
    return()
  endif(TARGET ${TGT})

  message(FATAL_ERROR "${TGT} target does not exist")
endfunction(AssertTarget TGT)
