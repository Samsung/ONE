if(NOT BUILD_ONERT_LLM)
  return()
endif(NOT BUILD_ONERT_LLM)

list(APPEND ONERT_LLM_SRCS "src/onert_llm.cc")
list(APPEND ONERT_LLM_SRCS "src/args.cc")
list(APPEND ONERT_LLM_SRCS "src/nnfw_util.cc")
list(APPEND ONERT_LLM_SRCS "src/rawformatter.cc")

add_executable(onert_llm ${ONERT_LLM_SRCS})

target_include_directories(onert_llm PRIVATE src)

target_link_libraries(onert_llm arser)
target_link_libraries(onert_llm nnfw_common)
target_link_libraries(onert_llm nnfw-dev)

install(TARGETS onert_llm DESTINATION ${CMAKE_INSTALL_BINDIR})
