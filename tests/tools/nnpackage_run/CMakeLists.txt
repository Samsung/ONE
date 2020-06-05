if(NOT BUILD_NNPACKAGE_RUN)
  return()
endif(NOT BUILD_NNPACKAGE_RUN)

if(NOT BUILD_ONERT)
  return()
endif(NOT BUILD_ONERT)

nnfw_find_package(HDF5 COMPONENTS CXX QUIET)
if(NOT HDF5_FOUND)
  message(WARNING "HDF5 NOT found. Install libhdf5-dev or set EXT_HDF5_DIR to build nnpackage_run.")
  return()
endif(NOT HDF5_FOUND)

list(APPEND NNPACKAGE_RUN_SRCS "src/nnpackage_run.cc")
list(APPEND NNPACKAGE_RUN_SRCS "src/args.cc")
list(APPEND NNPACKAGE_RUN_SRCS "src/h5formatter.cc")
list(APPEND NNPACKAGE_RUN_SRCS "src/nnfw_util.cc")

nnfw_find_package(Boost REQUIRED)
nnfw_find_package(Ruy QUIET)

add_executable(nnpackage_run ${NNPACKAGE_RUN_SRCS})
target_include_directories(nnpackage_run PRIVATE src)
target_include_directories(nnpackage_run PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(nnpackage_run PRIVATE ${HDF5_INCLUDE_DIRS})

target_link_libraries(nnpackage_run onert_core onert tflite_loader)
target_link_libraries(nnpackage_run tensorflow-lite ${LIB_PTHREAD} dl nnfw_lib_tflite)
target_link_libraries(nnpackage_run nnfw-dev)
if(BUILD_BOOST)
  # We have to use Boost::program_options, instead of boost_program_optinos
  # Boost::program_options provides the full path for our own built boost.
  target_link_libraries(nnpackage_run Boost::program_options)
else()
  # We cannot use `Boost::program_options` on aarch64
  # because it uses boost 1.65.1, which requires cmake >= 3.9.3
  # to identify Boost::program_options imported target correctly,
  # However, it uses cmake 3.5. (old version).
  target_link_libraries(nnpackage_run boost_program_options)
endif()
target_link_libraries(nnpackage_run ${HDF5_CXX_LIBRARIES})
target_link_libraries(nnpackage_run nnfw_lib_benchmark)
if(Ruy_FOUND AND PROFILE_RUY)
  target_link_libraries(nnpackage_run ruy_instrumentation)
  target_link_libraries(nnpackage_run ruy_profiler)
endif(Ruy_FOUND AND PROFILE_RUY)

install(TARGETS nnpackage_run DESTINATION bin)
