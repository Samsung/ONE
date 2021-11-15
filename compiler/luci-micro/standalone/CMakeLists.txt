cmake_minimum_required(VERSION 3.10)
project(luci_interpreter_micro_standalone)

# Add fake target, so nothing is build
set(BUILD_WHITELIST "dummy")

add_subdirectory(${NNAS_ROOT}/infra/nncc ${CMAKE_CURRENT_BINARY_DIR}/nncc)

set(ONE_COMPILER_SRC_DIR "${NNAS_PROJECT_SOURCE_DIR}/compiler")
include_directories(${ONE_COMPILER_SRC_DIR}/luci-micro/mbed-os)
nnas_find_package(FlatBuffersSource EXACT 1.10 QUIET)

include_directories(${FlatBuffersSource_DIR}/include)

add_subdirectory(${ONE_COMPILER_SRC_DIR}/loco ${CMAKE_CURRENT_BINARY_DIR}/loco)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/angkor ${CMAKE_CURRENT_BINARY_DIR}/angkor)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/oops ${CMAKE_CURRENT_BINARY_DIR}/oops)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/pepper-str ${CMAKE_CURRENT_BINARY_DIR}/pepper-str)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/logo ${CMAKE_CURRENT_BINARY_DIR}/logo)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/logo-core ${CMAKE_CURRENT_BINARY_DIR}/logo-core)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/hermes-std ${CMAKE_CURRENT_BINARY_DIR}/hermes-std)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/hermes ${CMAKE_CURRENT_BINARY_DIR}/hermes)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/pepper-strcast ${CMAKE_CURRENT_BINARY_DIR}/pepper-strcast)

add_subdirectory(${ONE_COMPILER_SRC_DIR}/locomotiv ${CMAKE_CURRENT_BINARY_DIR}/locomotiv)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/luci/lang ${CMAKE_CURRENT_BINARY_DIR}/luci/lang)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/luci/import ${CMAKE_CURRENT_BINARY_DIR}/luci/import)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/luci/profile ${CMAKE_CURRENT_BINARY_DIR}/luci/profile)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/luci/env ${CMAKE_CURRENT_BINARY_DIR}/luci/env)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/luci/plan ${CMAKE_CURRENT_BINARY_DIR}/luci/plan)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/luci/log ${CMAKE_CURRENT_BINARY_DIR}/luci/log)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/luci/logex ${CMAKE_CURRENT_BINARY_DIR}/luci/logex)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/locop ${CMAKE_CURRENT_BINARY_DIR}/locop)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/pp ${CMAKE_CURRENT_BINARY_DIR}/pp)

add_subdirectory(${ONE_COMPILER_SRC_DIR}/luci-interpreter ${CMAKE_CURRENT_BINARY_DIR}/luci-interpreter)
