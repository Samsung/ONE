nnas_find_package(HDF5 COMPONENTS STATIC QUIET)

if(NOT HDF5_FOUND)
  message(STATUS "Build minmax-embedder: FAILED (missing HDF5)")
  return()
endif(NOT HDF5_FOUND)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(minmax-embedder "${SOURCES}")
target_include_directories(minmax-embedder PUBLIC ${HDF5_INCLUDE_DIRS})

target_link_libraries(minmax-embedder ${HDF5_CXX_LIBRARIES})
target_link_libraries(minmax-embedder safemain)
target_link_libraries(minmax-embedder loco)
target_link_libraries(minmax-embedder luci_import)
target_link_libraries(minmax-embedder luci_service)
target_link_libraries(minmax-embedder luci_pass)
target_link_libraries(minmax-embedder luci_export)
target_link_libraries(minmax-embedder luci_env)
target_link_libraries(minmax-embedder arser)
target_link_libraries(minmax-embedder vconone)

install(TARGETS minmax-embedder DESTINATION bin)
