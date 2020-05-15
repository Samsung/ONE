
set(SRCS_EVAL_TESTER
      src/EvalTester.cpp
   )

add_executable(luci_interpreter_tester ${SRCS_EVAL_TESTER})
target_link_libraries(luci_interpreter_tester PRIVATE luci_import)
target_link_libraries(luci_interpreter_tester PRIVATE luci_lang)
target_link_libraries(luci_interpreter_tester PRIVATE luci_interpreter)
