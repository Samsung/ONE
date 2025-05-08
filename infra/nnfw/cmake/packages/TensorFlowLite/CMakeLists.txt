# Reference: https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/lite/CMakeLists.txt
#
# Tensorflow Lite library 2.16.1
#
set(TFLITE_SOURCE_DIR ${TensorFlowSource_DIR}/tensorflow/lite)
set(TSL_SOURCE_DIR "${TensorFlowSource_DIR}/third_party/xla/third_party/tsl")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate TensorFlow Lite FlatBuffer code.
# We used to have an actual compilation logic with flatc but decided to use
# schema_generated.h since flatc doesn't work with cross compilation.
set(TFLITE_FLATBUFFERS_SCHEMA_DIR "${TFLITE_SOURCE_DIR}/schema")

macro(populate_source_vars SOURCE_DIR SOURCES_VAR)
  cmake_parse_arguments(ARGS "RECURSE" "" "FILTER" ${ARGN})
  if(ARGS_RECURSE)
    set(GLOB_OP GLOB_RECURSE)
  else()
    set(GLOB_OP GLOB)
  endif()
  set(DEFAULT_FILE_FILTER ".*(_test|test_util)\\.(c|cc|h)$")
  file(${GLOB_OP} FOUND_SOURCES "${SOURCE_DIR}/*.*")
  list(FILTER FOUND_SOURCES INCLUDE REGEX ".*\\.(c|cc|h)$")
  list(FILTER FOUND_SOURCES EXCLUDE REGEX "${DEFAULT_FILE_FILTER}")
  foreach(FILE_FILTER ${ARGS_FILTER})
    list(FILTER FOUND_SOURCES EXCLUDE REGEX "${FILE_FILTER}")
  endforeach()
  list(APPEND ${SOURCES_VAR} ${FOUND_SOURCES})
endmacro()
# Simplifies inclusion of non-test sources and headers from a directory
# relative to TFLITE_SOURCE_DIR. See populate_source_vars() for the
# description of arguments including and following SOURCES_VAR.
macro(populate_tflite_source_vars RELATIVE_DIR SOURCES_VAR)
  populate_source_vars(
    "${TFLITE_SOURCE_DIR}/${RELATIVE_DIR}" ${SOURCES_VAR} ${ARGN}
  )
endmacro()

# Additional compiler flags used when compiling TF Lite.
set(TFLITE_TARGET_PUBLIC_OPTIONS "-DEIGEN_NEON_GEBP_NR=4")

# Build a list of source files to compile into the TF Lite library.
populate_tflite_source_vars("." TFLITE_SRCS)

# This particular file is excluded because the more explicit approach to enable
# XNNPACK delegate is preferred to the weak-symbol one.
list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*tflite_with_xnnpack\\.cc$")

# Exclude Flex related files.
list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*with_selected_ops\\.cc$")

# Exclude tensorflow_profiler_logger files.
list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*tensorflow_profiler_logger\\.cc$")

# Use MMAP
list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*mmap_allocation_disabled\\.cc$")

# Handle TFLite logging source.
list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*minimal_logging_.*\\.cc$")
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Android")
  list(APPEND TFLITE_SRCS ${TFLITE_SOURCE_DIR}/minimal_logging_android.cc)
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "iOS")
  list(APPEND TFLITE_SRCS ${TFLITE_SOURCE_DIR}/minimal_logging_ios.cc)
else()
  list(APPEND TFLITE_SRCS ${TFLITE_SOURCE_DIR}/minimal_logging_default.cc)
endif()

populate_tflite_source_vars("core" TFLITE_CORE_SRCS)
populate_tflite_source_vars("core/acceleration/configuration" TFLITE_CORE_ACCELERATION_SRCS)
populate_tflite_source_vars("core/api" TFLITE_CORE_API_SRCS)
populate_tflite_source_vars("core/async" TFLITE_CORE_ASYNC_SRCS)
populate_tflite_source_vars("core/async/c" TFLITE_CORE_ASYNC_C_SRCS)
populate_tflite_source_vars("core/async/interop" TFLITE_CORE_ASYNC_INTEROP_SRCS)
populate_tflite_source_vars("core/async/interop/c" TFLITE_CORE_ASYNC_INTEROP_C_SRCS)
populate_tflite_source_vars("core/c" TFLITE_CORE_C_SRCS)
populate_tflite_source_vars("core/experimental/acceleration/configuration" TFLITE_CORE_EXPERIMENTAL_SRCS)
populate_tflite_source_vars("core/kernels" TFLITE_CORE_KERNELS_SRCS)
populate_tflite_source_vars("core/tools" TFLITE_CORE_TOOLS_SRCS)
populate_tflite_source_vars("c" TFLITE_C_SRCS)
populate_tflite_source_vars("delegates" TFLITE_DELEGATES_SRCS)

# Enable NNAPI
populate_tflite_source_vars("delegates/nnapi"
  TFLITE_DELEGATES_NNAPI_SRCS
  FILTER "(_test_list|_disabled)\\.(cc|h)$"
)
populate_tflite_source_vars(
  "nnapi" TFLITE_NNAPI_SRCS FILTER "(_disabled)\\.(cc|h)$"
)
list(APPEND TFLITE_NNAPI_SRCS
  "${TFLITE_SOURCE_DIR}/nnapi/sl/SupportLibrary.cc"
)

# Enable experimental delegate
populate_tflite_source_vars("delegates/external"
  TFLITE_DELEGATES_EXTERNAL_SRCS
  FILTER ".*(_test|_tester)\\.(cc|h)"
)

populate_tflite_source_vars("experimental/remat"
  TFLITE_EXPERIMENTAL_REMAT_SRCS
  FILTER ".*_test\\.(cc|h)$"
)

# Enable  experimental support for resource (need for build success)
populate_tflite_source_vars("experimental/resource"
TFLITE_EXPERIMENTAL_RESOURCE_SRCS
)

# Disable XNNPack

populate_tflite_source_vars("experimental/ruy"
  TFLITE_EXPERIMENTAL_RUY_SRCS
  FILTER
  ".*(test(_fast|_slow|_special_specs))\\.(cc|h)$"
  ".*(benchmark|tune_tool|example)\\.(cc|h)$"
)
populate_tflite_source_vars("experimental/ruy/profiler"
  TFLITE_EXPERIMENTAL_RUY_PROFILER_SRCS
  FILTER ".*(test|test_instrumented_library)\\.(cc|h)$"
)
# Enable Ruy
list(APPEND TFLITE_TARGET_PUBLIC_OPTIONS "-DTFLITE_WITH_RUY")

populate_tflite_source_vars("kernels"
  TFLITE_KERNEL_SRCS
  FILTER "(.*_test_util_internal|test_.*|.*_ops_wrapper)\\.(cc|h)"
)
populate_tflite_source_vars("kernels/internal" TFLITE_KERNEL_INTERNAL_SRCS)
populate_tflite_source_vars("kernels/internal/optimized"
  TFLITE_KERNEL_INTERNAL_OPT_SRCS
)
populate_tflite_source_vars("kernels/internal/optimized/integer_ops"
  TFLITE_KERNEL_INTERNAL_OPT_INTEGER_OPS_SRCS
)
populate_tflite_source_vars("kernels/internal/optimized/sparse_ops"
  TFLITE_KERNEL_INTERNAL_OPT_SPARSE_OPS_SRCS
)
populate_tflite_source_vars("kernels/internal/reference"
  TFLITE_KERNEL_INTERNAL_REF_SRCS
)
populate_tflite_source_vars("kernels/internal/reference/integer_ops"
  TFLITE_KERNEL_INTERNAL_REF_INTEGER_OPS_SRCS
)
populate_tflite_source_vars("kernels/internal/reference/sparse_ops"
  TFLITE_KERNEL_INTERNAL_REF_SPARSE_OPS_SRCS
)
populate_tflite_source_vars("kernels/internal/optimized/4bit"
  TFLITE_KERNEL_INTERNAL_OPT_4BIT_SRCS
  FILTER "(.*neon.*|.*sse.*)\\.(cc|h)"
)
set(TFLITE_PROFILER_SRCS
  ${TFLITE_SOURCE_DIR}/profiling/platform_profiler.cc
  ${TFLITE_SOURCE_DIR}/profiling/root_profiler.h
  ${TFLITE_SOURCE_DIR}/profiling/root_profiler.cc
  ${TFLITE_SOURCE_DIR}/profiling/telemetry/profiler.cc
  ${TFLITE_SOURCE_DIR}/profiling/telemetry/profiler.h
  ${TFLITE_SOURCE_DIR}/profiling/telemetry/telemetry.cc
  ${TFLITE_SOURCE_DIR}/profiling/telemetry/c/telemetry_setting_internal.cc
  ${TFLITE_SOURCE_DIR}/profiling/telemetry/c/telemetry_setting_internal.h
  ${TFLITE_SOURCE_DIR}/profiling/telemetry/c/profiler.h
  ${TFLITE_SOURCE_DIR}/profiling/telemetry/c/telemetry_setting.h
  ${TFLITE_SOURCE_DIR}/profiling/telemetry/telemetry_status.h
)
if(CMAKE_SYSTEM_NAME MATCHES "Android")
  list(APPEND TFLITE_PROFILER_SRCS
    ${TFLITE_SOURCE_DIR}/profiling/atrace_profiler.cc
  )
endif()

# include headers
list(APPEND TFLITE_INCLUDE_DIRS "${TSL_SOURCE_DIR}")
list(APPEND TFLITE_INCLUDE_DIRS "${TensorFlowSource_DIR}")
list(APPEND TFLITE_INCLUDE_DIRS "${TensorFlowGEMMLowpSource_DIR}")
list(APPEND TFLITE_INCLUDE_DIRS "${Fp16Source_DIR}/include")
#list(APPEND TFLITE_INCLUDE_DIRS "${Pybind11Source_DIR}/include")
list(APPEND TFLITE_INCLUDE_DIRS "${CpuInfoSource_DIR}")
list(APPEND TFLITE_INCLUDE_DIRS "${MLDtypesSource_DIR}")

if(NEON2SSESource_FOUND)
  list(APPEND TFLITE_INCLUDE_DIRS "${NEON2SSESource_DIR}")
endif(NEON2SSESource_FOUND)

# TFLite library
set(_ALL_TFLITE_SRCS
  ${TFLITE_CORE_ACCELERATION_SRCS}
  ${TFLITE_CORE_API_SRCS}
  ${TFLITE_CORE_C_SRCS}
  ${TFLITE_CORE_EXPERIMENTAL_SRCS}
  ${TFLITE_CORE_KERNELS_SRCS}
  ${TFLITE_CORE_SRCS}
  ${TFLITE_CORE_ASYNC_SRCS}
  ${TFLITE_CORE_ASYNC_C_SRCS}
  ${TFLITE_CORE_ASYNC_INTEROP_SRCS}
  ${TFLITE_CORE_ASYNC_INTEROP_C_SRCS}
  ${TFLITE_CORE_TOOLS_SRCS}
  ${TFLITE_C_SRCS}
  ${TFLITE_DELEGATES_FLEX_SRCS}
  ${TFLITE_DELEGATES_GPU_SRCS}
  ${TFLITE_DELEGATES_NNAPI_SRCS}
  ${TFLITE_DELEGATES_SRCS}
  ${TFLITE_DELEGATES_XNNPACK_SRCS}
  ${TFLITE_DELEGATES_EXTERNAL_SRCS}
  ${TFLITE_EXPERIMENTAL_REMAT_SRCS}
  ${TFLITE_EXPERIMENTAL_RESOURCE_SRCS}
  ${TFLITE_EXPERIMENTAL_RUY_PROFILER_SRCS}
  ${TFLITE_EXPERIMENTAL_RUY_SRCS}
  ${TFLITE_KERNEL_INTERNAL_OPT_INTEGER_OPS_SRCS}
  ${TFLITE_KERNEL_INTERNAL_OPT_SPARSE_OPS_SRCS}
  ${TFLITE_KERNEL_INTERNAL_OPT_SRCS}
  ${TFLITE_KERNEL_INTERNAL_OPT_4BIT_SRCS}
  ${TFLITE_KERNEL_INTERNAL_REF_INTEGER_OPS_SRCS}
  ${TFLITE_KERNEL_INTERNAL_REF_SPARSE_OPS_SRCS}
  ${TFLITE_KERNEL_INTERNAL_REF_SRCS}
  ${TFLITE_KERNEL_INTERNAL_SRCS}
  ${TFLITE_KERNEL_SRCS}
  ${TFLITE_NNAPI_SRCS}
  ${TFLITE_SRCS}
  ${TFLITE_PROFILER_SRCS}
  ${TFLITE_SOURCE_DIR}/internal/signature_def.h
  ${TFLITE_SOURCE_DIR}/kernels/internal/utils/sparsity_format_converter.cc
  ${TFLITE_SOURCE_DIR}/schema/conversion_metadata_generated.h
  ${TFLITE_SOURCE_DIR}/schema/schema_utils.cc
  ${TFLITE_SOURCE_DIR}/schema/schema_generated.h
)
add_library(tensorflow-lite-2.16.1 STATIC
  ${_ALL_TFLITE_SRCS}
)
target_include_directories(tensorflow-lite-2.16.1
  SYSTEM PUBLIC
    ${TFLITE_INCLUDE_DIRS}
)

target_compile_options(tensorflow-lite-2.16.1 PUBLIC ${TFLITE_TARGET_PUBLIC_OPTIONS})
set_property(TARGET tensorflow-lite-2.16.1 PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(tensorflow-lite-2.16.1 eigen flatbuffers::flatbuffers-23.5.26 ruy abseil farmhash oourafft::fftsg2d pthreadpool ${LIB_PTHREAD} dl)
if(NOT ANDROID)
  target_link_libraries(tensorflow-lite-2.16.1 rt)
endif()

# Define TF_LITE_DISABLE_X86_NEON for debug build
# If we upgrade NEON2SSE version, we can remove below line
if(NEON2SSESource_FOUND)
  target_compile_definitions(tensorflow-lite-2.16.1 PRIVATE $<$<CONFIG:Debug>:TF_LITE_DISABLE_X86_NEON>)
endif(NEON2SSESource_FOUND)

if(ANDROID)
  target_link_libraries(tensorflow-lite-2.16.1 log)
  #target_include_directories(tensorflow-lite-2.16.1 PUBLIC "${NDK_DIR}/..")
endif()
