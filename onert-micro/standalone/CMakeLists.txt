cmake_minimum_required(VERSION 3.10)
project(luci_interpreter_micro_standalone)

# Add fake target, so nothing is build
set(BUILD_WHITELIST "dummy")

add_subdirectory(${NNAS_ROOT}/infra/nncc ${CMAKE_CURRENT_BINARY_DIR}/nncc)

nnas_find_package(FlatBuffersSource EXACT 2.0 QUIET)
include_directories(${FlatBuffersSource_DIR}/include)

# TODO: fix luci/plan for new luci-micro without luci/IR
add_subdirectory(${NNAS_PROJECT_SOURCE_DIR}/runtime-lite/luci-interpreter ${CMAKE_CURRENT_BINARY_DIR}/luci-interpreter)
