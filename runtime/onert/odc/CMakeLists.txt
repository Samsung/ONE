if(CMAKE_CROSSCOMPILING)
  return()
endif()

nnfw_find_package(Luci QUIET)
if(NOT Luci_FOUND)
  message(STATUS "Luci not found. Skip onert_odc")
  return()
endif()

file(GLOB_RECURSE SOURCES "*.cc")

add_library(onert_odc SHARED ${SOURCES})
target_link_libraries(onert_odc PRIVATE onert_core luci::import luci::export luci::pass luci::loco)

install(TARGETS onert_odc LIBRARY DESTINATION lib)
