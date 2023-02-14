set(INTERPRETER src/CircleInterpreter.cpp)

add_executable(circle-interpreter ${INTERPRETER})
target_link_libraries(circle-interpreter PRIVATE arser)
target_link_libraries(circle-interpreter PRIVATE loco)
target_link_libraries(circle-interpreter PRIVATE luci_import)
target_link_libraries(circle-interpreter PRIVATE luci_interpreter)
target_link_libraries(circle-interpreter PRIVATE safemain)
target_link_libraries(circle-interpreter PRIVATE vconone)

install(TARGETS circle-interpreter DESTINATION bin)
