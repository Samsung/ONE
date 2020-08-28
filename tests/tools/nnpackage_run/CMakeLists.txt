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

if(HDF5_FOUND)
  set(NNPACKAGE_RUN nnpackage_run_random)
  set(NNPACKAGE_RUN_WITH_HDF5 nnpackage_run)
else(HDF5_FOUND)
  message(WARNING "HDF5 NOT found. Install libhdf5-dev or set EXT_HDF5_DIR to support load/dump in nnpackage_run.")
  set(NNPACKAGE_RUN nnpackage_run)
endif(HDF5_FOUND)

add_executable(${NNPACKAGE_RUN} ${NNPACKAGE_RUN_SRCS})

if(Ruy_FOUND AND PROFILE_RUY)
  list(APPEND NNPACKAGE_RUN_LINK_LIB ruy_instrumentation)
  list(APPEND NNPACKAGE_RUN_LINK_LIB ruy_profiler)
endif(Ruy_FOUND AND PROFILE_RUY)

list(APPEND NNPACKAGE_RUN_INCLUDE_DIR src ${Boost_INCLUDE_DIRS})
list(APPEND NNPACKAGE_RUN_LINK_LIB onert_core onert tflite_loader)
list(APPEND NNPACKAGE_RUN_LINK_LIB nnfw_lib_tflite jsoncpp)
list(APPEND NNPACKAGE_RUN_LINK_LIB nnfw-dev)
list(APPEND NNPACKAGE_RUN_LINK_LIB ${Boost_PROGRAM_OPTIONS_LIBRARY})
list(APPEND NNPACKAGE_RUN_LINK_LIB nnfw_lib_benchmark)

target_include_directories(${NNPACKAGE_RUN} PRIVATE ${NNPACKAGE_RUN_INCLUDE_DIR})
target_link_libraries(${NNPACKAGE_RUN} ${NNPACKAGE_RUN_LINK_LIB})

install(TARGETS ${NNPACKAGE_RUN} DESTINATION bin)

if(HDF5_FOUND)
  list(APPEND NNPACKAGE_RUN_HDF5_SRCS "src/h5formatter.cc")

  add_executable(${NNPACKAGE_RUN_WITH_HDF5} ${NNPACKAGE_RUN_SRCS} ${NNPACKAGE_RUN_HDF5_SRCS})
  target_compile_definitions(${NNPACKAGE_RUN_WITH_HDF5} PRIVATE ONERT_HAVE_HDF5=1)
  target_include_directories(${NNPACKAGE_RUN_WITH_HDF5} PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(${NNPACKAGE_RUN_WITH_HDF5} ${HDF5_CXX_LIBRARIES})

  target_include_directories(${NNPACKAGE_RUN_WITH_HDF5} PRIVATE ${NNPACKAGE_RUN_INCLUDE_DIR})
  target_link_libraries(${NNPACKAGE_RUN_WITH_HDF5} ${NNPACKAGE_RUN_LINK_LIB})

  install(TARGETS ${NNPACKAGE_RUN_WITH_HDF5} DESTINATION bin)
endif(HDF5_FOUND)

