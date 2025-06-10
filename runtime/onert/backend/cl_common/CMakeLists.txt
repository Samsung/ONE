file(GLOB_RECURSE SOURCES "src/*.cc")

add_library(${LIB_ONERT_BACKEND_CL_COMMON} STATIC ${SOURCES})

target_include_directories(${LIB_ONERT_BACKEND_CL_COMMON} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(${LIB_ONERT_BACKEND_CL_COMMON} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(${LIB_ONERT_BACKEND_CL_COMMON} PUBLIC onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_CL_COMMON} PRIVATE nnfw_common nnfw_coverage)
