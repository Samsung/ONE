if(NOT BUILD_NNPACKAGE_RUN)
  return()
endif(NOT BUILD_NNPACKAGE_RUN)

if(NOT BUILD_ONERT)
  return()
endif(NOT BUILD_ONERT)

find_package(HDF5 COMPONENTS CXX QUIET)
if(NOT HDF5_FOUND)
  message(WARNING "HDF5 NOT found. Install libhdf5-dev to build nnpackage_run.")
  return()
endif(NOT HDF5_FOUND)

# Append missing libaec, which is required by libsz, which is required by libhdf5.
list(APPEND HDF5_CXX_LIBRARIES aec)

list(APPEND NNPACKAGE_RUN_SRCS "src/nnpackage_run.cc")
list(APPEND NNPACKAGE_RUN_SRCS "src/args.cc")
list(APPEND NNPACKAGE_RUN_SRCS "src/h5formatter.cc")
list(APPEND NNPACKAGE_RUN_SRCS "src/nnfw_util.cc")

nnas_find_package(Boost REQUIRED)

add_executable(nnpackage_run ${NNPACKAGE_RUN_SRCS})
target_include_directories(nnpackage_run PRIVATE src)
target_include_directories(nnpackage_run PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(nnpackage_run PRIVATE ${HDF5_INCLUDE_DIRS})

target_link_libraries(nnpackage_run onert_core onert tflite_loader)
target_link_libraries(nnpackage_run tensorflow-lite ${LIB_PTHREAD} dl nnfw_lib_tflite)
target_link_libraries(nnpackage_run nnfw-dev)
target_link_libraries(nnpackage_run boost_program_options)
target_link_libraries(nnpackage_run ${HDF5_CXX_LIBRARIES})
target_link_libraries(nnpackage_run nnfw_lib_benchmark)

install(TARGETS nnpackage_run DESTINATION bin)
