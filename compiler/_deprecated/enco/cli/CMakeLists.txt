file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(enco-cli ${SOURCES})
target_include_directories(enco-cli PRIVATE src)
target_link_libraries(enco-cli enco_intf_cmdline)
target_link_libraries(enco-cli enco_intf_frontend)
target_link_libraries(enco-cli enco_core)
target_link_libraries(enco-cli dl)
# Let's use project-wide compile options
target_link_libraries(enco-cli nncc_common)
