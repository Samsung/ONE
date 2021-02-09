# Copyright (c) 2021 ARM Limited. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.19.0)
set(LUCI_MICRO ON CACHE BOOL "LUCI_MICRO BUILD FLAG")
set(MBED_PATH ${CMAKE_CURRENT_SOURCE_DIR}/mbed-os CACHE INTERNAL "")
set(MBED_CONFIG_PATH ${CMAKE_CURRENT_SOURCE_DIR}/.mbedbuild CACHE INTERNAL "")
set(APP_TARGET luci-micro)
set(LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib CACHE INTERNAL "")

set(NNAS_PROJECT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE
  INTERNAL "Where to find nnas top-level source directory"
)

set(NNAS_EXTERNALS_DIR
  "${NNAS_PROJECT_SOURCE_DIR}/externals" CACHE
  INTERNAL "Where to download external dependencies"
)
set(NNFW_OVERLAY_DIR "${CMAKE_BINARY_DIR}/overlay" CACHE
    INTERNAL "Where locally built external dependencies are installed")

# Share package build script with compiler
set(EXT_OVERLAY_DIR ${NNFW_OVERLAY_DIR})

# This allows find_package to access configurations installed inside overlay
list(APPEND CMAKE_PREFIX_PATH "${EXT_OVERLAY_DIR}")

macro(nnas_include PREFIX)
  include("${NNAS_PROJECT_SOURCE_DIR}/infra/cmake/modules/${PREFIX}.cmake")
endmacro(nnas_include)

# Runtime 'find_package()' wrapper to find in cmake/packages folder
#
# Example:
#  nnfw_find_package(Boost): Load settings from 'BoostConfig.cmake' file
#    - this may drop warnings like "-- Could NOT find Boost (missing: Boost_DIR)
#  nnfw_find_package(Boost QUIET): Load settings silently, without warnings
#  nnfw_find_package(Boost REQUIRED): Load settings but stop with error when failed
macro(nnfw_find_package PREFIX)
  find_package(${PREFIX} CONFIG NO_DEFAULT_PATH
    PATHS ${CMAKE_SOURCE_DIR}/cmake/packages
    ${ARGN}
  )
endmacro(nnfw_find_package)

# Common 'find_package()' wrapper to find in infra/cmake/packages folder
macro(nnas_find_package PREFIX)
  find_package(${PREFIX} CONFIG NO_DEFAULT_PATH
    PATHS ${NNAS_PROJECT_SOURCE_DIR}/infra/cmake/packages
    ${ARGN}
  )
endmacro(nnas_find_package)

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_EXTENSIONS OFF)

include(${MBED_PATH}/tools/cmake/app.cmake)
SET(GCC_COVERAGE_COMPILE_FLAGS "-fexceptions")
SET(GCC_COVERAGE_LINK_FLAGS    "")

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

include_directories(PUBLIC ${LIB_PATH}/resources)
include_directories(PUBLIC ${LIB_PATH}/third_party/flatbuffers/include)
include_directories(PUBLIC ${LIB_PATH}/third_party)
include_directories(PUBLIC ${LIB_PATH}/luci/log/include)



add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

add_subdirectory(${LIB_PATH}/stdex)
add_subdirectory(${LIB_PATH}/oops)
add_subdirectory(${LIB_PATH}/pepper-str)
add_subdirectory(${LIB_PATH}/logo)
add_subdirectory(${LIB_PATH}/logo-core)
add_subdirectory(${LIB_PATH}/loco)
add_subdirectory(${LIB_PATH}/locomotiv)
add_subdirectory(${LIB_PATH}/angkor)
add_subdirectory(${LIB_PATH}/luci/lang)
add_subdirectory(${LIB_PATH}/luci/import)
add_subdirectory(${LIB_PATH}/luci/env)


add_subdirectory(${MBED_PATH})

add_executable(${APP_TARGET}
    main.cpp
)

mbed_configure_app_target(${APP_TARGET})

#mbed_set_mbed_target_linker_script(${APP_TARGET})

project(${APP_TARGET})

target_link_libraries(${APP_TARGET} mbed-os)
target_link_libraries(${APP_TARGET} stdex)
target_link_libraries(${APP_TARGET} oops)
target_link_libraries(${APP_TARGET} pepper_str)
target_link_libraries(${APP_TARGET} logo)
target_link_libraries(${APP_TARGET} logo_core)
target_link_libraries(${APP_TARGET} loco)
target_link_libraries(${APP_TARGET} locomotiv)
target_link_libraries(${APP_TARGET} angkor)
target_link_libraries(${APP_TARGET} luci_lang)
target_link_libraries(${APP_TARGET} luci_import)
target_link_libraries(${APP_TARGET} luci_env)

mbed_set_post_build(${APP_TARGET})

option(VERBOSE_BUILD "Have a verbose build process")
if(VERBOSE_BUILD)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif()
