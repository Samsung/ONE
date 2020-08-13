set(LIB_ONERT_BACKEND_CPU onert_backend_cpu)
set(LIB_ONERT_BACKEND_CPU_OPS onert_backend_cpu_ops)

nnfw_find_package(Ruy REQUIRED)

file(GLOB OPS_SOURCES "ops/*.cc")
file(GLOB SOURCES "*.cc")

add_library(${LIB_ONERT_BACKEND_CPU_OPS} STATIC ${OPS_SOURCES})
target_link_libraries(${LIB_ONERT_BACKEND_CPU_OPS} PRIVATE nnfw_lib_cker)
target_link_libraries(${LIB_ONERT_BACKEND_CPU_OPS} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_CPU_OPS} PRIVATE ruy)
target_link_libraries(${LIB_ONERT_BACKEND_CPU_OPS} PRIVATE ruy_instrumentation)
target_link_libraries(${LIB_ONERT_BACKEND_CPU_OPS} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_CPU_OPS} PRIVATE nnfw_coverage)
set_target_properties(${LIB_ONERT_BACKEND_CPU_OPS} PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(${LIB_ONERT_BACKEND_CPU} SHARED ${SOURCES})
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE ${LIB_ONERT_BACKEND_CPU_OPS})
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE nnfw_lib_cker)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE nnfw_coverage)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE ruy)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} INTERFACE ruy_instrumentation)

set_target_properties(${LIB_ONERT_BACKEND_CPU} PROPERTIES OUTPUT_NAME backend_cpu)

install(TARGETS ${LIB_ONERT_BACKEND_CPU} DESTINATION lib)
