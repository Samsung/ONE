set(LIB_ONERT_BACKEND_CPU onert_backend_cpu)

nnfw_find_package(Ruy REQUIRED)

file(GLOB_RECURSE SOURCES "*.cc")

add_library(${LIB_ONERT_BACKEND_CPU} SHARED ${SOURCES})

target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE nnfw_lib_cker)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE nnfw_coverage)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE ruy)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} INTERFACE ruy_instrumentation)

set_target_properties(${LIB_ONERT_BACKEND_CPU} PROPERTIES OUTPUT_NAME backend_cpu)

add_custom_command(TARGET ${LIB_ONERT_BACKEND_CPU} POST_BUILD
                   COMMAND ${CMAKE_STRIP} "--strip-unneeded" $<TARGET_FILE_NAME:${LIB_ONERT_BACKEND_CPU}>)

install(TARGETS ${LIB_ONERT_BACKEND_CPU} DESTINATION lib)
