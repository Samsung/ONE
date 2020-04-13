if(NOT BUILD_NEURUN)
  return()
endif(NOT BUILD_NEURUN)

add_subdirectory(backend)
add_subdirectory(frontend)
add_subdirectory(core)
add_subdirectory(api)
add_subdirectory(sample)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

add_subdirectory(test)
