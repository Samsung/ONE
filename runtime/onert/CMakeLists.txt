if(NOT BUILD_ONERT)
  return()
endif(NOT BUILD_ONERT)

add_subdirectory(backend)
add_subdirectory(frontend)
add_subdirectory(core)
add_subdirectory(api)
add_subdirectory(sample)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

add_subdirectory(test)
