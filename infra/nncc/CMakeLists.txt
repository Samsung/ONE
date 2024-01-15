cmake_minimum_required(VERSION 3.10)

project(nncc)

enable_testing()

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(NNAS_PROJECT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE
  INTERNAL "Where to find nnas top-level source directory"
)

set(NNAS_EXTERNALS_DIR
  "${NNAS_PROJECT_SOURCE_DIR}/externals" CACHE
  INTERNAL "Where to download external dependencies"
)
set(NNCC_OVERLAY_DIR "${CMAKE_BINARY_DIR}/overlay" CACHE
    INTERNAL "Where locally built external dependencies are installed")

# Share package build script with runtime
set(EXT_OVERLAY_DIR ${NNCC_OVERLAY_DIR})

# This allows find_package to access configurations installed inside overlay
list(APPEND CMAKE_PREFIX_PATH "${EXT_OVERLAY_DIR}")

macro(nnas_include PREFIX)
  include("${NNAS_PROJECT_SOURCE_DIR}/infra/cmake/modules/${PREFIX}.cmake")
endmacro(nnas_include)

macro(nnas_find_package PREFIX)
  find_package(${PREFIX}
               CONFIG NO_DEFAULT_PATH
               PATHS ${NNAS_PROJECT_SOURCE_DIR}/infra/cmake/packages
               ${ARGN})
endmacro(nnas_find_package)

macro(nnas_find_package_folder PREFIX FIND_FOLDER)
  find_package(${PREFIX}
               CONFIG NO_DEFAULT_PATH
               PATHS ${NNAS_PROJECT_SOURCE_DIR}/infra/cmake/packages ${FIND_FOLDER}
               ${ARGN})
endmacro(nnas_find_package_folder)

# nncc_find_resource(NAME) will update the following variables
#
#   NAME_FOUND
#   NAME_DIR
#
# TODO Explain how to add a resource in README.md
function(nncc_find_resource NAME)
  set(RESOURCE_DIR "${NNAS_PROJECT_SOURCE_DIR}/res/${NAME}")

  if(NOT IS_DIRECTORY ${RESOURCE_DIR})
    set(${NAME}_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT IS_DIRECTORY ${RESOURCE_DIR})

  set(${NAME}_DIR ${RESOURCE_DIR} PARENT_SCOPE)
  set(${NAME}_FOUND TRUE PARENT_SCOPE)
endfunction(nncc_find_resource)

###
### CMake configuration
###
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Type of build" FORCE)
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "Use '${CMAKE_BUILD_TYPE}' configuration")

# Prefer -pthread to -lpthread for find_package(Threads ...)
#
# std::thread code compiled only with -lpthread emits the following runtime error (on GCC 4.8.4)
#
#   terminate called after throwing an instance of 'std::system_error'
#     what():  Enable multithreading to use std::thread: Operation not permitted
#
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

# identify platform: HOST_PLATFORM, TARGET_PLATFORM and related
# note: this should be placed before flags and options setting
nnas_include(IdentifyPlatform)

# Configuration flags
include("cmake/CfgOptionFlags.cmake")

nnas_find_package(GTest QUIET)

option(ENABLE_TEST "Build Tests using Google Test" ${GTest_FOUND})

if(${ENABLE_TEST} AND NOT ${GTest_FOUND})
  message(FATAL_ERROR "Google Test is required to enable test")
endif(${ENABLE_TEST} AND NOT ${GTest_FOUND})

option(ENABLE_COVERAGE "Build for coverage test" OFF)
if(${ENABLE_COVERAGE} AND NOT ${ENABLE_TEST})
  message(FATAL_ERROR "Test should be enabled to measure test coverage")
endif(${ENABLE_COVERAGE} AND NOT ${ENABLE_TEST})

if(${ENABLE_TEST})
  include(CTest)
endif(${ENABLE_TEST})

# apply compilation flags
# NOTE this should be after all option
include("cmake/ApplyCompileFlags.cmake")

###
### Target
###
add_library(nncc_common INTERFACE)
if(ENABLE_STRICT_BUILD)
  # TODO Remove -Wno-reoder
  target_compile_options(nncc_common INTERFACE -Werror -Wall -Wextra -Wno-reorder)
endif(ENABLE_STRICT_BUILD)

add_library(nncc_coverage INTERFACE)
if(ENABLE_COVERAGE)
  target_compile_options(nncc_coverage INTERFACE -g -O0 -fprofile-arcs -ftest-coverage)
  target_link_libraries(nncc_coverage INTERFACE gcov)
endif(ENABLE_COVERAGE)

###
### Function
###
# TODO Remove this nnas_include
nnas_include(OptionalTargetTools)
nnas_include(AddSubdirectories)

add_subdirectory("${NNAS_PROJECT_SOURCE_DIR}/compiler" "${CMAKE_BINARY_DIR}/compiler")
