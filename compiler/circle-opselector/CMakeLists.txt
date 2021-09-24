set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(circle-opselector ${DRIVER} ${SOURCES})
target_include_directories(circle-opselector PRIVATE src)
target_link_libraries(circle-opselector foder)
target_link_libraries(circle-opselector safemain)
target_link_libraries(circle-opselector loco)
target_link_libraries(circle-opselector luci_import)
target_link_libraries(circle-opselector luci_export)
target_link_libraries(circle-opselector arser)
target_link_libraries(circle-opselector vconone)
target_link_libraries(circle-opselector luci_service)
target_link_libraries(circle-opselector luci_profile)

install(TARGETS circle-opselector DESTINATION bin)
