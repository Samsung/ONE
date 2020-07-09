cmake_minimum_required(VERSION 3.1)

project(nncc)

enable_testing()

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# This feature works with CMake 3.5.2 or later. However, using previous versions does not produce
# an error. We are still officially using CMake 3.1.0, but put this code for the sake of semantic
# support in various development tools.
# Todo: Someday, CMake needs to be updated to 3.7.2 or later to take advantage of improvements
#       such as `cmake-server`.
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
  find_package(${PREFIX} CONFIG NO_DEFAULT_PATH
    PATHS ${NNAS_PROJECT_SOURCE_DIR}/infra/cmake/packages
    ${ARGN}
  )
endmacro(nnas_find_package)

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

###
### Configuration
###
option(DOWNLOAD_PROTOBUF "Download Protocol Buffer source" ON)
option(BUILD_PROTOBUF "Locally build Protocol Buffer from the downloaded source" ON)
option(DOWNLOAD_EIGEN "Download Eigen source" ON)
option(DOWNLOAD_FARMHASH "Download farmhash source" ON)
option(DOWNLOAD_GEMMLOWP "Download GEMM low precesion library source" ON)
option(DOWNLOAD_NEON2SSE "Download NEON2SSE library source" ON)
option(DOWNLOAD_GFLAGS "Download GFlags source" OFF)
option(DOWNLOAD_FLATBUFFERS "Download FlatBuffers source" ON)
option(BUILD_FLATBUFFERS "Locally build Flatbuffers from the downloaded source" ON)
option(DOWNLOAD_TENSORFLOW "Download TensorFlow source" ON)
option(DOWNLOAD_CAFFE "Download Caffe source" ON)
option(DOWNLOAD_PYTORCH "Download Pytorch source" ON)
option(DOWNLOAD_ONNX "Download ONNX source" ON)
option(DOWNLOAD_ABSEIL "Download Abseil-cpp source" ON)

option(DOWNLOAD_GTEST "Download Google Test source" ON)
option(BUILD_GTEST "Build Google Test from the downloaded source" ON)
option(DOWNLOAD_HDF5 "Download HDF5 source" ON)
option(BUILD_HDF5 "Build HDF5 from the downloaded source" ON)

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

option(ENABLE_STRICT_BUILD "Treat warning as error" OFF)

# This option might be turned ON for Windows native build.
# Check our ProtobufConfig.cmake for its usage.
option(USE_PROTOBUF_LEGACY_IMPORT "Use legacy MODULE mode import rather than CONFIG mode" OFF)

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
