# TODO Find how to test logging framework
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(luci_log ${LIBRARY_TYPE} ${SOURCES})
target_include_directories(luci_log PUBLIC include)
target_link_libraries(luci_log PUBLIC hermes)
target_link_libraries(luci_log PRIVATE hermes_std)
target_link_libraries(luci_log PRIVATE nncc_common)
target_link_libraries(luci_log PRIVATE luci_env)
install(TARGETS luci_log DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")
