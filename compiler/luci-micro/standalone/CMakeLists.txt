cmake_minimum_required(VERSION 3.10)
project(luci_interpreter_micro_standalone)

# Add fake target, so nothing is build
set(BUILD_WHITELIST "dummy")

add_subdirectory(${NNAS_ROOT}/infra/nncc ${CMAKE_CURRENT_BINARY_DIR}/nncc)

set(ONE_COMPILER_SRC_DIR "${NNAS_PROJECT_SOURCE_DIR}/compiler")
nnas_find_package(FlatBuffersSource EXACT 2.0 QUIET)

include_directories(${FlatBuffersSource_DIR}/include)

add_subdirectory(${ONE_COMPILER_SRC_DIR}/mio-circle04 ${CMAKE_CURRENT_BINARY_DIR}/mio-circle04)
# TODO: fix luci/plan for new luci-micro without luci/IR
# add_subdirectory(${ONE_COMPILER_SRC_DIR}/luci/plan ${CMAKE_CURRENT_BINARY_DIR}/luci/plan)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/luci-micro/luci-interpreter ${CMAKE_CURRENT_BINARY_DIR}/luci-interpreter)
