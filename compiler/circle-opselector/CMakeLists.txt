set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "pass/*.cpp")

add_executable(circle-opselector ${DRIVER} ${SOURCES})
target_include_directories(circle-opselector PRIVATE pass)
target_link_libraries(circle-opselector foder)
target_link_libraries(circle-opselector safemain)
target_link_libraries(circle-opselector loco)
target_link_libraries(circle-opselector luci_import)
target_link_libraries(circle-opselector luci_export)
target_link_libraries(circle-opselector arser)
target_link_libraries(circle-opselector vconone)

install(TARGETS circle-opselector DESTINATION bin)
