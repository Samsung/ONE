if(NOT BUILD_TFLITE_VANILLA_RUN)
  return()
endif()

if(NOT BUILD_TENSORFLOW_LITE_2_8_0)
  set(BUILD_TENSORFLOW_LITE_2_8_0 ON)
  set(BUILD_TENSORFLOWRUY ON)
endif()

nnfw_find_package(TensorFlowLite EXACT 2.8.0 REQUIRED)
nnfw_find_package(Boost REQUIRED program_options)

list(APPEND TFLITE_RUN_SRCS "src/tflite_vanilla_run.cc")
list(APPEND TFLITE_RUN_SRCS "src/args.cc")

add_executable(tflite_vanilla_run ${TFLITE_RUN_SRCS})
target_include_directories(tflite_vanilla_run PRIVATE src)
target_include_directories(tflite_vanilla_run PRIVATE ${Boost_INCLUDE_DIRS})

target_link_libraries(tflite_vanilla_run tensorflow-lite-2.8.0 ${LIB_PTHREAD} dl)
target_link_libraries(tflite_vanilla_run ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(tflite_vanilla_run nnfw_lib_benchmark nnfw_lib_misc)

install(TARGETS tflite_vanilla_run DESTINATION bin)
