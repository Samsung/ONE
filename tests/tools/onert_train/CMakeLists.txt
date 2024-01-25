if(NOT BUILD_ONERT_TRAIN)
  return()
endif(NOT BUILD_ONERT_TRAIN)

if(NOT BUILD_ONERT)
  return()
endif(NOT BUILD_ONERT)

# Use C++17 for onert_train
# TODO: Remove this when we use C++17 for all runtime directories
set(CMAKE_CXX_STANDARD 17)

list(APPEND ONERT_TRAIN_SRCS "src/onert_train.cc")
list(APPEND ONERT_TRAIN_SRCS "src/args.cc")
list(APPEND ONERT_TRAIN_SRCS "src/nnfw_util.cc")
list(APPEND ONERT_TRAIN_SRCS "src/randomgen.cc")
list(APPEND ONERT_TRAIN_SRCS "src/rawformatter.cc")
list(APPEND ONERT_TRAIN_SRCS "src/rawdataloader.cc")
list(APPEND ONERT_TRAIN_SRCS "src/metrics.cc")

nnfw_find_package(Boost REQUIRED program_options)
nnfw_find_package(HDF5 QUIET)

if (HDF5_FOUND)
  list(APPEND ONERT_TRAIN_SRCS "src/h5formatter.cc")
endif()

add_executable(onert_train ${ONERT_TRAIN_SRCS})

if (HDF5_FOUND)
  target_compile_definitions(onert_train PRIVATE ONERT_HAVE_HDF5=1)
  target_include_directories(onert_train PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(onert_train ${HDF5_CXX_LIBRARIES})
else()
  message(WARNING "HDF5 NOT found. Install libhdf5-dev or set EXT_HDF5_DIR to support load/dump in onert_train.")
endif(HDF5_FOUND)

target_include_directories(onert_train PRIVATE src)
target_include_directories(onert_train PRIVATE ${Boost_INCLUDE_DIRS})

target_link_libraries(onert_train nnfw_lib_tflite jsoncpp)
target_link_libraries(onert_train nnfw-dev)
target_link_libraries(onert_train ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(onert_train nnfw_lib_benchmark)

install(TARGETS onert_train DESTINATION bin)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Unit Tests
set(TEST_ONERT_TRAIN test_onert_train)

file(GLOB_RECURSE ONERT_TRAIN_TEST_SRCS "test/*.cc")
list(APPEND ONERT_TRAIN_TEST_SRCS "src/rawdataloader.cc")
list(APPEND ONERT_TRAIN_TEST_SRCS "src/nnfw_util.cc")

add_executable(${TEST_ONERT_TRAIN} ${ONERT_TRAIN_TEST_SRCS})

target_link_libraries(${TEST_ONERT_TRAIN} nnfw-dev)
target_link_libraries(${TEST_ONERT_TRAIN} gtest gtest_main dl ${LIB_PTHREAD})

add_test(${TEST_ONERT_TRAIN} ${TEST_ONERT_TRAIN})
install(TARGETS ${TEST_ONERT_TRAIN} DESTINATION unittest)
