if(NOT BUILD_TFLITE_RUN)
  return()
endif(NOT BUILD_TFLITE_RUN)

list(APPEND TFLITE_RUN_SRCS "src/tflite_run.cc")
list(APPEND TFLITE_RUN_SRCS "src/args.cc")
list(APPEND TFLITE_RUN_SRCS "src/tensor_dumper.cc")
list(APPEND TFLITE_RUN_SRCS "src/tensor_loader.cc")

nnfw_find_package(Boost REQUIRED)

add_executable(tflite_run ${TFLITE_RUN_SRCS})
target_include_directories(tflite_run PRIVATE src)
target_include_directories(tflite_run PRIVATE ${Boost_INCLUDE_DIRS})

target_link_libraries(tflite_run tensorflow-lite ${LIB_PTHREAD} dl nnfw_lib_tflite)
if(BUILD_BOOST)
  # We have to use Boost::program_options, instead of boost_program_optinos
  # Boost::program_options provides the full path for our own built boost.
  target_link_libraries(tflite_run Boost::program_options)
else()
  # We cannot use `Boost::program_options` on aarch64
  # because it uses boost 1.65.1, which requires cmake >= 3.9.3
  # to identify Boost::program_options imported target correctly,
  # However, it uses cmake 3.5. (old version).
  target_link_libraries(tflite_run boost_program_options)
endif()

target_link_libraries(tflite_run nnfw_lib_benchmark)

install(TARGETS tflite_run DESTINATION bin)

# TEST BUILD
nnfw_find_package(GTest)

if(NOT GTest_FOUND)
  return()
endif(NOT GTest_FOUND)

## Add test cpp file
add_executable(tflite_test src/tflite_test.cc)
## Link test executable against gtest & gtest_main
target_link_libraries(tflite_test gtest gtest_main ${LIB_PTHREAD})
## install test binary for packaging
install(TARGETS tflite_test DESTINATION unittest)
