# Unsupported architecture
nnfw_find_package(ARMCompute QUIET)
if(NOT ARMCompute_FOUND)
  return()
endif(NOT ARMCompute_FOUND)

set(LIB_ONERT_BACKEND_ACL_CL onert_backend_acl_cl)

file(GLOB SOURCES "*.cc")
file(GLOB OPERAND_SOURCES "operand/*.cc")
list(APPEND SOURCES ${OPERAND_SOURCES})
macro(OP NAME)
  # ACL_CL backend doesn't have separate operation files,
  # but we include the macro for consistency with other backend pattern
endmacro(OP)
include(Operation.lst)

add_library(${LIB_ONERT_BACKEND_ACL_CL} SHARED ${SOURCES})

target_link_libraries(${LIB_ONERT_BACKEND_ACL_CL} PRIVATE ${LIB_ONERT_BACKEND_ACL_COMMON})
target_link_libraries(${LIB_ONERT_BACKEND_ACL_CL} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_ACL_CL} PRIVATE nnfw_coverage)

# Set install rpath to find onert_core and acl library
set_target_properties(${LIB_ONERT_BACKEND_ACL_CL} PROPERTIES
  OUTPUT_NAME backend_acl_cl
  INSTALL_RPATH ${ONERT_RPATH_PLUGIN})

install(TARGETS ${LIB_ONERT_BACKEND_ACL_CL} DESTINATION ${ONERT_INSTALL_BACKENDDIR})
