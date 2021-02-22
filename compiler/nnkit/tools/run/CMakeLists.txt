if(NOT TARGET nnkit_intf_action)
  return()
endif(NOT TARGET nnkit_intf_action)

if(NOT TARGET nnkit_intf_backend)
  return()
endif(NOT TARGET nnkit_intf_backend)

if(NOT TARGET nnkit_support_cmdline)
  return()
endif(NOT TARGET nnkit_support_cmdline)

if(NOT TARGET nnkit_support_backend)
  return()
endif(NOT TARGET nnkit_support_backend)

add_executable(nnkit-run nnkit-run.cpp)
target_link_libraries(nnkit-run nnkit_intf_action)
target_link_libraries(nnkit-run nnkit_intf_backend)
target_link_libraries(nnkit-run nnkit_support_cmdline)
target_link_libraries(nnkit-run nnkit_support_backend)
