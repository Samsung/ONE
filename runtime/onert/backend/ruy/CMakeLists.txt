set(LIB_ONERT_BACKEND_RUY onert_backend_ruy)

nnfw_find_package(Ruy REQUIRED)

file(GLOB SOURCES "*.cc")
list(APPEND SOURCES ops/OperationUtils.cc)
macro(OP NAME)
  list(APPEND SOURCES ops/${NAME}Layer.cc)
endmacro(OP)
include(Operation.lst)

add_library(${LIB_ONERT_BACKEND_RUY} SHARED ${SOURCES})

target_link_libraries(${LIB_ONERT_BACKEND_RUY} PRIVATE nnfw_lib_ruy)
target_link_libraries(${LIB_ONERT_BACKEND_RUY} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_RUY} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_RUY} PRIVATE nnfw_coverage)
target_link_libraries(${LIB_ONERT_BACKEND_RUY} PRIVATE ruy)

set_target_properties(${LIB_ONERT_BACKEND_RUY} PROPERTIES
  OUTPUT_NAME backend_ruy
  INSTALL_RPATH ${ONERT_RPATH_PLUGIN})

install(TARGETS ${LIB_ONERT_BACKEND_RUY} DESTINATION ${ONERT_INSTALL_BACKENDDIR})
