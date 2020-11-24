set(LIB_ONERT_BACKEND_ACL_COMMON onert_backend_acl_common)

add_subdirectory(cpu)
add_subdirectory(acl_cl)
#add_subdirectory(acl_neon)
add_subdirectory(acl_common)
add_subdirectory(ruy)
add_subdirectory(xnnpack)
