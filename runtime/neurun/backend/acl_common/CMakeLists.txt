# Unsupported architecture
nnas_find_package(ARMCompute QUIET)
if(NOT ARMCompute_FOUND)
  return()
endif(NOT ARMCompute_FOUND)

file(GLOB SOURCES "*.cc")

add_library(${LIB_NEURUN_BACKEND_ACL_COMMON} STATIC ${SOURCES})

target_include_directories(${LIB_NEURUN_BACKEND_ACL_COMMON} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${LIB_NEURUN_BACKEND_ACL_COMMON} PUBLIC neurun_core)
target_link_libraries(${LIB_NEURUN_BACKEND_ACL_COMMON} PUBLIC arm_compute arm_compute_ex)
target_link_libraries(${LIB_NEURUN_BACKEND_ACL_COMMON} PUBLIC nnfw_lib_misc)
target_link_libraries(${LIB_NEURUN_BACKEND_ACL_COMMON} PRIVATE nnfw_common)
target_link_libraries(${LIB_NEURUN_BACKEND_ACL_COMMON} PRIVATE nnfw_coverage)

set_target_properties(${LIB_NEURUN_BACKEND_ACL_COMMON} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${LIB_NEURUN_BACKEND_ACL_COMMON} PROPERTIES OUTPUT_NAME backend_acl_common)
