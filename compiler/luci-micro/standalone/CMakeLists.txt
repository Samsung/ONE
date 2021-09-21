cmake_minimum_required(VERSION 3.10)
project(luci_interpreter_micro_standalone)

# Add fake target, so nothing is build
set(BUILD_WHITELIST "dummy")

add_subdirectory(${NNAS_ROOT}/infra/nncc ${CMAKE_CURRENT_BINARY_DIR}/nncc)

set(ONE_COMPILER_SRC_DIR "${NNAS_PROJECT_SOURCE_DIR}/compiler")

add_subdirectory(${ONE_COMPILER_SRC_DIR}/loco ${CMAKE_CURRENT_BINARY_DIR}/loco)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/angkor ${CMAKE_CURRENT_BINARY_DIR}/angkor)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/oops ${CMAKE_CURRENT_BINARY_DIR}/oops)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/pepper-str ${CMAKE_CURRENT_BINARY_DIR}/pepper-str)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/logo ${CMAKE_CURRENT_BINARY_DIR}/logo)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/logo-core ${CMAKE_CURRENT_BINARY_DIR}/logo-core)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/locomotiv ${CMAKE_CURRENT_BINARY_DIR}/locomotiv)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/luci/lang ${CMAKE_CURRENT_BINARY_DIR}/luci/lang)

add_subdirectory(${ONE_COMPILER_SRC_DIR}/luci-interpreter ${CMAKE_CURRENT_BINARY_DIR}/luci-interpreter)
