if(NOT BUILD_NNPACKAGE_RUN)
  return()
endif(NOT BUILD_NNPACKAGE_RUN)

if(NOT BUILD_ONERT)
  return()
endif(NOT BUILD_ONERT)

list(APPEND NNPACKAGE_RUN_SRCS "src/nnpackage_run.cc")
list(APPEND NNPACKAGE_RUN_SRCS "src/args.cc")
list(APPEND NNPACKAGE_RUN_SRCS "src/nnfw_util.cc")
list(APPEND NNPACKAGE_RUN_SRCS "src/randomgen.cc")

nnfw_find_package(Boost REQUIRED program_options)
nnfw_find_package(Ruy QUIET)
nnfw_find_package(HDF5 QUIET)

if (HDF5_FOUND)
  list(APPEND NNPACKAGE_RUN_SRCS "src/h5formatter.cc")
endif()

add_executable(nnpackage_run ${NNPACKAGE_RUN_SRCS})

if (HDF5_FOUND)
  target_compile_definitions(nnpackage_run PRIVATE ONERT_HAVE_HDF5=1)
  target_include_directories(nnpackage_run PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(nnpackage_run ${HDF5_CXX_LIBRARIES})
else()
  message(WARNING "HDF5 NOT found. Install libhdf5-dev or set EXT_HDF5_DIR to support load/dump in nnpackage_run.")
endif(HDF5_FOUND)

target_include_directories(nnpackage_run PRIVATE src)
target_include_directories(nnpackage_run PRIVATE ${Boost_INCLUDE_DIRS})

target_link_libraries(nnpackage_run nnfw_lib_tflite jsoncpp)
target_link_libraries(nnpackage_run nnfw-dev)
target_link_libraries(nnpackage_run ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(nnpackage_run nnfw_lib_benchmark)
if(Ruy_FOUND AND PROFILE_RUY)
  target_link_libraries(nnpackage_run ruy_instrumentation)
  target_link_libraries(nnpackage_run ruy_profiler)
endif(Ruy_FOUND AND PROFILE_RUY)

install(TARGETS nnpackage_run DESTINATION bin)
