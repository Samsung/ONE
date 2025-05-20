set(LIB_ONERT_BACKEND_TRIX onert_backend_trix)

find_package(PkgConfig REQUIRED)
# TRIXEngine version is required to higher than 2.5.0
pkg_check_modules(TRIXEngine QUIET IMPORTED_TARGET npu-engine>2.5.0)

if(NOT TRIXEngine_FOUND)
  message(STATUS "ONERT backend: Failed to find TRIXEngine")
  return()
endif(NOT TRIXEngine_FOUND)
message(STATUS "ONERT backend: Found TRIXEngine")

file(GLOB_RECURSE SOURCES "*.cc")

add_library(${LIB_ONERT_BACKEND_TRIX} SHARED ${SOURCES})

target_link_libraries(${LIB_ONERT_BACKEND_TRIX} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_TRIX} PRIVATE PkgConfig::TRIXEngine)
target_link_libraries(${LIB_ONERT_BACKEND_TRIX} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_TRIX} PRIVATE nnfw_coverage)

set_target_properties(${LIB_ONERT_BACKEND_TRIX} PROPERTIES
  OUTPUT_NAME backend_trix
  INSTALL_RPATH "$ORIGIN:$ORIGIN/..")

install(TARGETS ${LIB_ONERT_BACKEND_TRIX} DESTINATION lib/nnfw/backend)
