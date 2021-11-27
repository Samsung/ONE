set(SRCS_EVAL_TESTER
      src/EvalDriver.cpp
      src/sample_model.h
      src/sample_model.cpp
   )

add_executable(model_eval_verifier ${SRCS_EVAL_TESTER})
target_link_libraries(model_eval_verifier PRIVATE oops)
target_link_libraries(model_eval_verifier PRIVATE loco)
target_link_libraries(model_eval_verifier PRIVATE luci_import)
target_link_libraries(model_eval_verifier PRIVATE luci_export)
target_link_libraries(model_eval_verifier PRIVATE luci_lang)
target_link_libraries(model_eval_verifier PRIVATE luci_interpreter)
target_link_libraries(model_eval_verifier PRIVATE safemain)

install(TARGETS model_eval_verifier DESTINATION bin)
