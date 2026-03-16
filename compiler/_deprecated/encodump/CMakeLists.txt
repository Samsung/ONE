if(NOT TARGET enco_intf_frontend)
  return()
endif(NOT TARGET enco_intf_frontend)

if(NOT TARGET enco_core)
  return()
endif(NOT TARGET enco_core)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(encodump ${SOURCES})
target_include_directories(encodump PRIVATE src)
target_link_libraries(encodump enco_intf_frontend)
target_link_libraries(encodump enco_core)
target_link_libraries(encodump safemain)
target_link_libraries(encodump dl)
