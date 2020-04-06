set(RUY_BASE ${RuySource_DIR}/ruy)

#
# Ruy library
#
file(GLOB RUY_CORE_SRCS "${RUY_BASE}/*.cc")
file(GLOB RUY_CORE_TESTS "${RUY_BASE}/*test*.cc")
list(REMOVE_ITEM RUY_CORE_SRCS ${RUY_CORE_TESTS})

list(APPEND RUY_SRCS ${RUY_CORE_SRCS})
list(APPEND RUY_SRCS "${RUY_BASE}/profiler/instrumentation.cc")
list(REMOVE_ITEM RUY_SRCS "${RUY_BASE}/benchmark.cc")
list(REMOVE_ITEM RUY_SRCS "${RUY_BASE}/example.cc")
list(REMOVE_ITEM RUY_SRCS "${RUY_BASE}/example_advanced.cc")
list(REMOVE_ITEM RUY_SRCS "${RUY_BASE}/tune_tool.cc")
list(REMOVE_ITEM RUY_SRCS "${RUY_BASE}/pmu.cc")

list(APPEND RUY_INCLUDES "${RuySource_DIR}")

add_library(ruy STATIC ${RUY_SRCS})
target_include_directories(ruy SYSTEM PUBLIC ${RUY_INCLUDES})
set_property(TARGET ruy PROPERTY POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

