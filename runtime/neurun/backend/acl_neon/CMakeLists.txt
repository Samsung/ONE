# Unsupported architecture
nnas_find_package(ARMCompute QUIET)
if(NOT ARMCompute_FOUND)
  return()
endif(NOT ARMCompute_FOUND)

set(LIB_NEURUN_BACKEND_ACL_NEON neurun_backend_acl_neon)

file(GLOB_RECURSE SOURCES "*.cc")

add_library(${LIB_NEURUN_BACKEND_ACL_NEON} SHARED ${SOURCES})

target_link_libraries(${LIB_NEURUN_BACKEND_ACL_NEON} PRIVATE ${LIB_NEURUN_BACKEND_ACL_COMMON})
target_link_libraries(${LIB_NEURUN_BACKEND_ACL_NEON} PRIVATE nnfw_common)
target_link_libraries(${LIB_NEURUN_BACKEND_ACL_NEON} PRIVATE nnfw_coverage)

set_target_properties(${LIB_NEURUN_BACKEND_ACL_NEON} PROPERTIES OUTPUT_NAME backend_acl_neon)

install(TARGETS ${LIB_NEURUN_BACKEND_ACL_NEON} DESTINATION lib)
