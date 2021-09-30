# TODO Find how to test logging-ex utility
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(luci_logex ${LIBRARY_TYPE} ${SOURCES})
target_include_directories(luci_logex PUBLIC include)
target_link_libraries(luci_logex PUBLIC loco)
target_link_libraries(luci_logex PUBLIC locop)
target_link_libraries(luci_logex PRIVATE luci_log)
target_link_libraries(luci_logex PRIVATE luci_lang)
target_link_libraries(luci_logex PRIVATE hermes_std)
target_link_libraries(luci_logex PRIVATE nncc_common)
target_link_libraries(luci_logex PRIVATE pepper_str)
install(TARGETS luci_logex DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")
