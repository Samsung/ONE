# Dummy target to build all nnc unit tests
add_custom_target(nnc_unit_tests)
add_custom_target(
  nnc_check
  COMMAND ${CMAKE_CTEST_COMMAND}
  DEPENDS nnc_unit_tests)

add_subdirectory(pass)
add_subdirectory(soft_backend)
add_subdirectory(acl_backend)
add_subdirectory(support)
add_subdirectory(optimizations)
add_subdirectory(transformations)
