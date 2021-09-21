set(SRCS_EVAL_TESTER
      src/EvalDriver.cpp
   )

add_executable(luci_eval_driver ${SRCS_EVAL_TESTER})
target_link_libraries(luci_eval_driver PRIVATE oops)
target_link_libraries(luci_eval_driver PRIVATE loco)
target_link_libraries(luci_eval_driver PRIVATE luci_micro_import)
target_link_libraries(luci_eval_driver PRIVATE luci_export)
target_link_libraries(luci_eval_driver PRIVATE luci_lang)
target_link_libraries(luci_eval_driver PRIVATE luci_interpreter)
target_link_libraries(luci_eval_driver PRIVATE safemain)

install(TARGETS luci_eval_driver DESTINATION bin)
