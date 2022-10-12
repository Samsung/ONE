if(NOT BUILD_ONERT_RUN)
  return()
endif(NOT BUILD_ONERT_RUN)

list(APPEND ONERT_RUN_SRCS "onert_run.cc")
list(APPEND ONERT_RUN_SRCS "args.cc")

nnfw_find_package(Boost REQUIRED program_options)

add_executable(onert_run ${ONERT_RUN_SRCS})
target_include_directories(onert_run PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(onert_run nnfw_lib_benchmark nnfw_lib_misc nnfw-dev ${Boost_PROGRAM_OPTIONS_LIBRARY})

install(TARGETS onert_run DESTINATION bin)
