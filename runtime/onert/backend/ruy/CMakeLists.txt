set(LIB_ONERT_BACKEND_RUY onert_backend_ruy)

nnfw_find_package(Ruy REQUIRED)

file(GLOB_RECURSE SOURCES "*.cc")

add_library(${LIB_ONERT_BACKEND_RUY} SHARED ${SOURCES})

target_link_libraries(${LIB_ONERT_BACKEND_RUY} PRIVATE nnfw_lib_ruy)
target_link_libraries(${LIB_ONERT_BACKEND_RUY} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_RUY} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_RUY} PRIVATE nnfw_coverage)
target_link_libraries(${LIB_ONERT_BACKEND_RUY} PRIVATE ruy)

set_target_properties(${LIB_ONERT_BACKEND_RUY} PROPERTIES
  OUTPUT_NAME backend_ruy
  INSTALL_RPATH "$ORIGIN:$ORIGIN/../..")

install(TARGETS ${LIB_ONERT_BACKEND_RUY} DESTINATION lib/nnfw/backend)
