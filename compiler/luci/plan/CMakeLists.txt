file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(luci_plan ${LIBRARY_TYPE} ${SOURCES})
target_include_directories(luci_plan PRIVATE src)
target_include_directories(luci_plan PUBLIC include)
target_link_libraries(luci_plan PUBLIC loco)
target_link_libraries(luci_plan PUBLIC luci_lang)

install(TARGETS luci_plan DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")
