if(NOT BUILD_TFLITE_RUN_2_2_0)
  return()
endif()

if(NOT BUILD_TENSORFLOW_LITE_2_2_0)
  set(BUILD_TENSORFLOW_LITE_2_2_0 ON)
endif()

nnfw_find_package(TensorFlowLite-2.2.0 REQUIRED)
nnfw_find_package(Boost REQUIRED)

list(APPEND TFLITE_RUN_SRCS "src/tflite_run_2_2_0.cc")
list(APPEND TFLITE_RUN_SRCS "src/args.cc")

add_executable(tflite_run_2_2_0 ${TFLITE_RUN_SRCS})
target_include_directories(tflite_run_2_2_0 PRIVATE src)
target_include_directories(tflite_run_2_2_0 PRIVATE ${Boost_INCLUDE_DIRS})

target_link_libraries(tflite_run_2_2_0 tensorflow-lite-2.2.0 ${LIB_PTHREAD} dl)
target_link_libraries(tflite_run_2_2_0 ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(tflite_run_2_2_0 nnfw_lib_benchmark nnfw_lib_misc)

install(TARGETS tflite_run_2_2_0 DESTINATION bin)