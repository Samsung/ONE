add_library(${LIB_ONERT_BACKEND_CL_COMMON} INTERFACE)

target_include_directories(${LIB_ONERT_BACKEND_CL_COMMON} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${LIB_ONERT_BACKEND_CL_COMMON} INTERFACE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_CL_COMMON} INTERFACE nnfw_lib_misc)
