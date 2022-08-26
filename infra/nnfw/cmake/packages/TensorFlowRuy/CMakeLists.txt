set(RUY_BASE ${TensorFlowRuySource_DIR}/ruy)

#
# Ruy library
#
file(GLOB RUY_CORE_SRCS "${RUY_BASE}/*.cc")
file(GLOB RUY_CORE_TESTS "${RUY_BASE}/*test*.cc")
list(REMOVE_ITEM RUY_CORE_SRCS ${RUY_CORE_TESTS})

list(APPEND RUY_SRCS ${RUY_CORE_SRCS})
list(REMOVE_ITEM RUY_SRCS "${RUY_BASE}/benchmark.cc")
list(REMOVE_ITEM RUY_SRCS "${RUY_BASE}/example.cc")
list(REMOVE_ITEM RUY_SRCS "${RUY_BASE}/example_advanced.cc")
list(REMOVE_ITEM RUY_SRCS "${RUY_BASE}/tune_tool.cc")
list(REMOVE_ITEM RUY_SRCS "${RUY_BASE}/pmu.cc")
list(REMOVE_ITEM RUY_SRCS "${RUY_BASE}/create_trmul_params.cc")
list(REMOVE_ITEM RUY_SRCS "${RUY_BASE}/prepare_packed_matrices.cc")

list(APPEND RUY_INSTRUMENTATION_SRCS "${RUY_BASE}/profiler/instrumentation.cc")

if(PROFILE_RUY)
  list(APPEND RUY_PROFILER_SRCS "${RUY_BASE}/profiler/profiler.cc")
  list(APPEND RUY_PROFILER_SRCS "${RUY_BASE}/profiler/treeview.cc")
endif(PROFILE_RUY)

list(APPEND RUY_INCLUDES "${TensorFlowRuySource_DIR}")

add_library(tensorflow-ruy STATIC ${RUY_SRCS})
target_include_directories(tensorflow-ruy SYSTEM PUBLIC ${RUY_INCLUDES})
target_compile_options(tensorflow-ruy PRIVATE -O3)

target_include_directories(tensorflow-ruy PRIVATE ${CpuInfoSource_DIR})
target_link_libraries(tensorflow-ruy PRIVATE cpuinfo)
target_compile_definitions(tensorflow-ruy PRIVATE RUY_HAVE_CPUINFO)

add_library(tensorflow-ruy_instrumentation ${RUY_INSTRUMENTATION_SRCS})
target_include_directories(tensorflow-ruy_instrumentation SYSTEM PUBLIC ${RUY_INCLUDES})
target_compile_options(tensorflow-ruy_instrumentation PRIVATE -O3)

set_target_properties(tensorflow-ruy tensorflow-ruy_instrumentation PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(PROFILE_RUY)
  add_library(tensorflow-ruy_profiler STATIC ${RUY_PROFILER_SRCS})
  target_include_directories(tensorflow-ruy_profiler SYSTEM PUBLIC ${RUY_INCLUDES})
  target_compile_options(tensorflow-ruy_profiler PRIVATE -O3)
  set_target_properties(tensorflow-ruy_profiler PROPERTIES POSITION_INDEPENDENT_CODE ON)

  target_compile_definitions(tensorflow-ruy PUBLIC RUY_PROFILER)
  target_compile_definitions(tensorflow-ruy_instrumentation PUBLIC RUY_PROFILER)
  target_compile_definitions(tensorflow-ruy_profiler PUBLIC RUY_PROFILER)
endif(PROFILE_RUY)
