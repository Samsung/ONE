set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "pass/*.cpp")

add_executable(opselector ${DRIVER} ${SOURCES})
target_include_directories(opselector PRIVATE pass)
target_link_libraries(opselector foder)
target_link_libraries(opselector safemain)
target_link_libraries(opselector loco)
target_link_libraries(opselector luci_import)
target_link_libraries(opselector luci_export)
target_link_libraries(opselector arser)
target_link_libraries(opselector vconone)

target_link_libraries(opselector luci_service)

install(TARGETS opselector DESTINATION bin)
