set(LIB_ONERT_BACKEND_GGML onert_backend_ggml)

file(GLOB SOURCES "*.cc")
macro(OP NAME)
  list(APPEND SOURCES ops/${NAME}Layer.cc)
endmacro(OP)
include(Operation.lst)

add_library(${LIB_ONERT_BACKEND_GGML} SHARED ${SOURCES})

target_link_libraries(${LIB_ONERT_BACKEND_GGML} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_GGML} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_GGML} PRIVATE nnfw_coverage)
target_link_libraries(${LIB_ONERT_BACKEND_GGML} PRIVATE ggml)

# Some used functions makes deprecated declarations warning
# TODO Resolve workaround
target_compile_options(${LIB_ONERT_BACKEND_GGML} PRIVATE -Wno-deprecated-declarations)

set_target_properties(${LIB_ONERT_BACKEND_GGML} PROPERTIES
  OUTPUT_NAME backend_ggml
  INSTALL_RPATH ${ONERT_RPATH_PLUGIN})

install(TARGETS ${LIB_ONERT_BACKEND_GGML} DESTINATION ${ONERT_INSTALL_BACKENDDIR})
