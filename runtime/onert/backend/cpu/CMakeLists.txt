nnfw_find_package(Ruy REQUIRED)

file(GLOB_RECURSE SOURCES "*.cc")

add_library(${LIB_ONERT_BACKEND_CPU} SHARED ${SOURCES})

target_include_directories(${LIB_ONERT_BACKEND_CPU} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE nnfw_lib_cker nnfw_lib_misc)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE nnfw_coverage)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE ruy)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} INTERFACE ruy_instrumentation)
target_link_libraries(${LIB_ONERT_BACKEND_CPU} PRIVATE ndarray)

set_target_properties(${LIB_ONERT_BACKEND_CPU} PROPERTIES OUTPUT_NAME backend_cpu)
set_target_properties(${LIB_ONERT_BACKEND_CPU} PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CMAKE_BUILD_TYPE_LC STREQUAL "release")
  add_custom_command(TARGET ${LIB_ONERT_BACKEND_CPU} POST_BUILD
                     COMMAND ${CMAKE_STRIP} "--strip-unneeded" $<TARGET_FILE_NAME:${LIB_ONERT_BACKEND_CPU}>)
endif()

install(TARGETS ${LIB_ONERT_BACKEND_CPU} DESTINATION lib)
