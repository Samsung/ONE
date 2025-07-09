nnfw_find_package(ARMCompute QUIET)
if(NOT ARMCompute_FOUND)
  return()
endif(NOT ARMCompute_FOUND)

function(add_kben_acl_neon_library)
  cmake_parse_arguments(ARG "" "NAME" "SOURCES" ${ARGN})

  add_library(${ARG_NAME} SHARED ${ARG_SOURCES})
  target_compile_options(${ARG_NAME} PRIVATE -Wno-psabi)
  target_include_directories(${ARG_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(${ARG_NAME} nonius)
  target_link_libraries(${ARG_NAME} arm_compute)
  target_link_libraries(${ARG_NAME} pthread)
  install(TARGETS ${ARG_NAME} DESTINATION ${CMAKE_INSTALL_LIBDIR}/kben)
endfunction(add_kben_acl_neon_library)

add_kben_acl_neon_library(NAME kben_acl_neon_conv SOURCES Convolution.cpp)
add_kben_acl_neon_library(NAME kben_acl_neon_transpose_conv SOURCES TransposeConv.cpp)
