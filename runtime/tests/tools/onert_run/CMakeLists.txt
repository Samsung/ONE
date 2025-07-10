if(NOT BUILD_ONERT_RUN)
  return()
endif(NOT BUILD_ONERT_RUN)

list(APPEND ONERT_RUN_SRCS "src/onert_run.cc")
list(APPEND ONERT_RUN_SRCS "src/args.cc")
list(APPEND ONERT_RUN_SRCS "src/nnfw_util.cc")
list(APPEND ONERT_RUN_SRCS "src/randomgen.cc")
list(APPEND ONERT_RUN_SRCS "src/rawformatter.cc")

nnfw_find_package(Ruy QUIET)
nnfw_find_package(HDF5 QUIET)

if (HDF5_FOUND)
  list(APPEND ONERT_RUN_SRCS "src/h5formatter.cc")
endif()

add_executable(onert_run ${ONERT_RUN_SRCS})

if (HDF5_FOUND)
  target_compile_definitions(onert_run PRIVATE ONERT_HAVE_HDF5=1)
  target_include_directories(onert_run PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(onert_run ${HDF5_CXX_LIBRARIES})
else()
  message(WARNING "HDF5 NOT found. Install libhdf5-dev or set EXT_HDF5_DIR to support load/dump in onert_run.")
endif(HDF5_FOUND)

target_include_directories(onert_run PRIVATE src)

target_link_libraries(onert_run nnfw_common)
target_link_libraries(onert_run nnfw_lib_tflite jsoncpp)
target_link_libraries(onert_run nnfw-dev)
target_link_libraries(onert_run arser)
target_link_libraries(onert_run nnfw_lib_benchmark)
if(Ruy_FOUND AND PROFILE_RUY)
  target_link_libraries(onert_run ruy_instrumentation)
  target_link_libraries(onert_run ruy_profiler)
endif(Ruy_FOUND AND PROFILE_RUY)

install(TARGETS onert_run DESTINATION ${CMAKE_INSTALL_BINDIR})
