set(LIB_ONERT_BACKEND_CPU onert_backend_cpu)

file(GLOB_RECURSE SOURCES "*.cc")

add_library(${LIB_ONERT_BACKEND_CPU} SHARED ${SOURCES})

target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE nnfw_lib_misc nnfw_lib_cker)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE nnfw_coverage)

set_target_properties(${LIB_ONERT_BACKEND_CPU} PROPERTIES OUTPUT_NAME backend_cpu)

install(TARGETS ${LIB_ONERT_BACKEND_CPU} DESTINATION lib)
