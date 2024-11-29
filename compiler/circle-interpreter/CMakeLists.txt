set(INTERPRETER
      src/CircleInterpreter.cpp
   )

add_executable(circle-interpreter ${INTERPRETER})
target_link_libraries(circle-interpreter PRIVATE arser)
target_link_libraries(circle-interpreter PRIVATE loco)
target_link_libraries(circle-interpreter PRIVATE luci_import)
target_link_libraries(circle-interpreter PRIVATE luci_interpreter)
target_link_libraries(circle-interpreter PRIVATE safemain)
target_link_libraries(circle-interpreter PRIVATE vconone)

install(TARGETS circle-interpreter DESTINATION bin)

set(INTERPRETER_CFFI
   src/CircleInterpreter_cffi.cpp
)

add_library(circle_interpreter_cffi SHARED ${INTERPRETER_CFFI})
target_link_libraries(circle_interpreter_cffi PRIVATE luci_import)
target_link_libraries(circle_interpreter_cffi PRIVATE luci_interpreter)

install(TARGETS circle_interpreter_cffi DESTINATION lib)
