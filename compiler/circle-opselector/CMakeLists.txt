set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "pass/*.cpp")

add_executable(circle_opselector ${DRIVER} ${SOURCES})
target_include_directories(circle_opselector PRIVATE pass)
target_link_libraries(circle_opselector foder)
target_link_libraries(circle_opselector safemain)
target_link_libraries(circle_opselector loco)
target_link_libraries(circle_opselector luci_import)
target_link_libraries(circle_opselector luci_export)
target_link_libraries(circle_opselector arser)
target_link_libraries(circle_opselector vconone)

target_link_libraries(circle_opselector luci_service)
target_link_libraries(circle_opselector luci_profile)

install(TARGETS circle_opselector DESTINATION bin)
