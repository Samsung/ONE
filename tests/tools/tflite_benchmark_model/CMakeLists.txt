if (NOT BUILD_TFLITE_BENCHMARK_MODEL)
  return()
endif(NOT BUILD_TFLITE_BENCHMARK_MODEL)

nnfw_find_package(TensorFlowLite EXACT 1.13.1 REQUIRED)

# TODO Remove this target_compile_definitions command, and just check its presence.
#      This change is prerequisites on pre-built tensorflow-lite package support
target_compile_definitions(tensorflow-lite PUBLIC "TFLITE_PROFILING_ENABLED")

file(GLOB_RECURSE SOURCES "*.cc")

nnas_find_package(TensorFlowSource EXACT 1.13.1 REQUIRED)
set(TENSORFLOW_LITE_BASE "${TensorFlowSource_DIR}/tensorflow/lite")
list(APPEND SOURCES "${TENSORFLOW_LITE_BASE}/tools/benchmark/benchmark_main.cc"
                    "${TENSORFLOW_LITE_BASE}/tools/benchmark/benchmark_model.cc"
                    "${TENSORFLOW_LITE_BASE}/tools/benchmark/benchmark_params.cc"
                    "${TENSORFLOW_LITE_BASE}/tools/benchmark/command_line_flags.cc")

add_executable(tflite_benchmark_model ${SOURCES})
target_compile_definitions(tflite_benchmark_model PUBLIC "TFLITE_PROFILING_ENABLED")
target_link_libraries(tflite_benchmark_model nnfw_lib_misc nnfw_lib_tflite nnfw_lib_profiling)
target_link_libraries(tflite_benchmark_model tensorflow-lite ${LIB_PTHREAD} dl)
install(TARGETS tflite_benchmark_model DESTINATION bin)
