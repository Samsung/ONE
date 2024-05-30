nnfw_find_package(TRIXEngine QUIET 2.5.0)

if(NOT TRIXEngine_FOUND)
  message(STATUS "NPUD backend: Failed to find TRIXEngine")
  return()
endif(NOT TRIXEngine_FOUND)
message(STATUS "NPUD backend: Found TRIXEngine")

file(GLOB_RECURSE SOURCES "*.cc")

add_library(npud_backend_trix SHARED ${SOURCES})

target_include_directories(npud_backend_trix PUBLIC ${NPUD_INCLUDE_DIRS})
target_link_libraries(npud_backend_trix PRIVATE nnfw_lib_misc)
target_link_libraries(npud_backend_trix PRIVATE trix_engine)

if(ENVVAR_NPUD_CONFIG)
  target_compile_definitions(npud_backend_trix PRIVATE ENVVAR_FOR_DEFAULT_CONFIG)
endif(ENVVAR_NPUD_CONFIG)

install(TARGETS npud_backend_trix DESTINATION lib)
