cmake_minimum_required(VERSION 3.10)
project(luci_interpreter_micro_standalone)

set(ONE_INFRA_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../infra CACHE INTERNAL "")
set(ONE_COMPILER_SRC_DIR ${NNAS_PROJECT_SOURCE_DIR}/compiler/ CACHE INTERNAL "")

macro(nnas_find_package PREFIX)
  message("nnas find package " " ${PREFIX} " ${ARGN})
  find_package(${PREFIX} CONFIG NO_DEFAULT_PATH
    PATHS ${NNAS_PROJECT_SOURCE_DIR}/infra/cmake/packages
    ${ARGN}
  )
endmacro(nnas_find_package)

macro(nnas_include PREFIX)
  include("${NNAS_PROJECT_SOURCE_DIR}/infra/cmake/modules/${PREFIX}.cmake")
endmacro(nnas_include)

set(DOWNLOAD_RUY ON)
set(DOWNLOAD_GEMMLOWP ON)
set(DOWNLOAD_EIGEN ON)
set(DOWNLOAD_TENSORFLOW ON)

#add_subdirectory(${ONE_INFRA_SRC_DIR}/nncc ${CMAKE_CURRENT_BINARY_DIR}/nncc)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/luci/lang ${CMAKE_CURRENT_BINARY_DIR}/luci/lang)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/loco ${CMAKE_CURRENT_BINARY_DIR}/loco)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/angkor ${CMAKE_CURRENT_BINARY_DIR}/angkor)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/oops ${CMAKE_CURRENT_BINARY_DIR}/oops)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/pepper-str ${CMAKE_CURRENT_BINARY_DIR}/pepper-str)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/logo ${CMAKE_CURRENT_BINARY_DIR}/logo)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/logo-core ${CMAKE_CURRENT_BINARY_DIR}/logo-core)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/locomotiv ${CMAKE_CURRENT_BINARY_DIR}/locomotiv)

add_subdirectory(${ONE_COMPILER_SRC_DIR}/luci-interpreter ${CMAKE_CURRENT_BINARY_DIR}/luci-interpreter)
