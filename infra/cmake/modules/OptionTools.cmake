function(envoption PREFIX DEFAULT_VALUE)
  set(VALUE ${DEFAULT_VALUE})

  if(DEFINED ENV{${PREFIX}})
    set(VALUE $ENV{${PREFIX}})
  endif()

  set(${PREFIX} ${VALUE} PARENT_SCOPE)
endfunction(envoption)
