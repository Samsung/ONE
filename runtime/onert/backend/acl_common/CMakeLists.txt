# Unsupported architecture
nnfw_find_package(ARMCompute QUIET)
if(NOT ARMCompute_FOUND)
  return()
endif(NOT ARMCompute_FOUND)

file(GLOB_RECURSE SOURCES "*.cc")

add_library(${LIB_ONERT_BACKEND_ACL_COMMON} STATIC ${SOURCES})

target_include_directories(${LIB_ONERT_BACKEND_ACL_COMMON} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${LIB_ONERT_BACKEND_ACL_COMMON} PUBLIC onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_ACL_COMMON} PUBLIC arm_compute arm_compute_ex)
target_link_libraries(${LIB_ONERT_BACKEND_ACL_COMMON} PUBLIC nnfw_lib_misc)
target_link_libraries(${LIB_ONERT_BACKEND_ACL_COMMON} PUBLIC ${LIB_ONERT_BACKEND_CL_COMMON})
target_link_libraries(${LIB_ONERT_BACKEND_ACL_COMMON} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_ACL_COMMON} PRIVATE nnfw_coverage)

set_target_properties(${LIB_ONERT_BACKEND_ACL_COMMON} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${LIB_ONERT_BACKEND_ACL_COMMON} PROPERTIES OUTPUT_NAME backend_acl_common)
