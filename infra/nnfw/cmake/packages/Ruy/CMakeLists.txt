set(RUY_BASE ${RuySource_DIR}/ruy)

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

list(APPEND RUY_INSTRUMENTATION_SRCS "${RUY_BASE}/profiler/instrumentation.cc")

if(PROFILE_RUY)
  list(APPEND RUY_PROFILER_SRCS "${RUY_BASE}/profiler/profiler.cc")
  list(APPEND RUY_PROFILER_SRCS "${RUY_BASE}/profiler/treeview.cc")
endif(PROFILE_RUY)

list(APPEND RUY_INCLUDES "${RuySource_DIR}")

add_library(ruy STATIC ${RUY_SRCS})
target_include_directories(ruy SYSTEM PUBLIC ${RUY_INCLUDES})
target_compile_options(ruy PRIVATE -O3)

target_include_directories(ruy PRIVATE ${CpuInfoSource_DIR})
target_link_libraries(ruy PRIVATE cpuinfo)
target_compile_definitions(ruy PRIVATE RUY_HAVE_CPUINFO)

add_library(ruy_instrumentation ${RUY_INSTRUMENTATION_SRCS})
target_include_directories(ruy_instrumentation SYSTEM PUBLIC ${RUY_INCLUDES})
target_compile_options(ruy_instrumentation PRIVATE -O3)

set_target_properties(ruy ruy_instrumentation PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(PROFILE_RUY)
  add_library(ruy_profiler STATIC ${RUY_PROFILER_SRCS})
  target_include_directories(ruy_profiler SYSTEM PUBLIC ${RUY_INCLUDES})
  target_compile_options(ruy_profiler PRIVATE -O3)
  set_target_properties(ruy_profiler PROPERTIES POSITION_INDEPENDENT_CODE ON)

  target_compile_definitions(ruy PUBLIC RUY_PROFILER)
  target_compile_definitions(ruy_instrumentation PUBLIC RUY_PROFILER)
  target_compile_definitions(ruy_profiler PUBLIC RUY_PROFILER)
endif(PROFILE_RUY)
