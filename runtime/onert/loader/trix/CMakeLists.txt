nnfw_find_package(TRIXEngine QUIET)
if(NOT TRIXEngine_FOUND)
  message(STATUS "ONERT frontend: Failed to find TRIXEngine")
  return()
endif(NOT TRIXEngine_FOUND)
message(STATUS "ONERT frontend: Found TRIXEngine")

list(APPEND SOURCES TrixLoader.cc)
# Loader library name should be lib<type>_loader.so
add_library(tvn_loader SHARED ${SOURCES})
set_target_properties(tvn_loader PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  INSTALL_RPATH ${ONERT_RPATH_PLUGIN})
target_link_libraries(tvn_loader PRIVATE onert_core)
target_link_libraries(tvn_loader PRIVATE nnfw_common nnfw_coverage)
target_link_libraries(tvn_loader PRIVATE trix-engine)

install(TARGETS tvn_loader DESTINATION ${ONERT_INSTALL_LOADERDIR})
