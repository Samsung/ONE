# Unsupported architecture
nnfw_find_package(ARMCompute QUIET)
if(NOT ARMCompute_FOUND)
  return()
endif(NOT ARMCompute_FOUND)

set(LIB_ONERT_BACKEND_ACL_CL onert_backend_acl_cl)

file(GLOB_RECURSE SOURCES "*.cc")

add_library(${LIB_ONERT_BACKEND_ACL_CL} SHARED ${SOURCES})

target_link_libraries(${LIB_ONERT_BACKEND_ACL_CL} PRIVATE ${LIB_ONERT_BACKEND_ACL_COMMON})
target_link_libraries(${LIB_ONERT_BACKEND_ACL_CL} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_ACL_CL} PRIVATE nnfw_coverage)

# Set install rpath to find onert_core and acl library
set_target_properties(${LIB_ONERT_BACKEND_ACL_CL} PROPERTIES
  OUTPUT_NAME backend_acl_cl
  INSTALL_RPATH ${ONERT_RPATH_PLUGIN})

install(TARGETS ${LIB_ONERT_BACKEND_ACL_CL} DESTINATION ${ONERT_INSTALL_BACKENDDIR})
