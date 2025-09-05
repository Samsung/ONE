if(NOT BUILD_GGMA_RUN)
  return()
endif(NOT BUILD_GGMA_RUN)

list(APPEND GGMA_RUN_SRCS "src/args.cc")
list(APPEND GGMA_RUN_SRCS "src/ggma_run.cc")
list(APPEND GGMA_RUN_SRCS "src/nnfw_util.cc")
list(APPEND GGMA_RUN_SRCS "src/rawformatter.cc")

add_executable(ggma_run ${GGMA_RUN_SRCS})

target_include_directories(ggma_run PRIVATE src)

target_link_libraries(ggma_run arser)
target_link_libraries(ggma_run ggma-dev)
target_link_libraries(ggma_run nnfw_common)
target_link_libraries(ggma_run nnfw-dev)

install(TARGETS ggma_run DESTINATION ${CMAKE_INSTALL_BINDIR})
