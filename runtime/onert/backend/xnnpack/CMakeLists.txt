set(LIB_ONERT_BACKEND_XNNPACK onert_backend_xnnpack)

# Unsupported architecture
nnfw_find_package(Xnnpack QUIET)
if(NOT Xnnpack_FOUND)
  return()
endif(NOT Xnnpack_FOUND)

file(GLOB_RECURSE SOURCES "*.cc")

add_library(${LIB_ONERT_BACKEND_XNNPACK} SHARED ${SOURCES})

target_link_libraries(${LIB_ONERT_BACKEND_XNNPACK} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_XNNPACK} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_XNNPACK} PRIVATE nnfw_coverage)
target_link_libraries(${LIB_ONERT_BACKEND_XNNPACK} PRIVATE pthreadpool)
target_link_libraries(${LIB_ONERT_BACKEND_XNNPACK} PRIVATE XNNPACK)

set_target_properties(${LIB_ONERT_BACKEND_XNNPACK} PROPERTIES OUTPUT_NAME backend_xnnpack)

install(TARGETS ${LIB_ONERT_BACKEND_XNNPACK} DESTINATION lib)
