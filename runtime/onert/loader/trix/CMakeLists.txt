if (NOT BUILD_TRIX_LOADER)
  return()
endif ()

nnfw_find_package(TRIXEngine QUIET 2.5.0)
if(TRIXEngine_FOUND)
  message(STATUS "ONERT frontend: Found TRIXEngine")
  list(APPEND SOURCES TrixLoader.cc)
else()
  message(STATUS "ONERT frontend: Failed to find TRIXEngine")
  return()
endif(TRIXEngine_FOUND)

# Loader library name should be lib<type>_loader.so
add_library(tvn_loader SHARED ${SOURCES})
set_target_properties(tvn_loader PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(tvn_loader PRIVATE onert_core)
target_link_libraries(tvn_loader PRIVATE nnfw_common nnfw_coverage)
target_link_libraries(tvn_loader PRIVATE trix_engine)

install(TARGETS tvn_loader DESTINATION lib)
