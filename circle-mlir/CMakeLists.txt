# Match the minimum required version of LLVM and MLIR
cmake_minimum_required(VERSION 3.13.4)

project(circle-mlir)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Type of build" FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options("-fexceptions")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/infra/cmake")

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

# configuration flags
include(CfgOptionFlags)

# enable test coverage
include(TestCoverage)

# enable ctest
include(CTest)

# enable googletest but do not install
set(INSTALL_GTEST OFF)
include(GTestHelper)
include(GoogleTest)

# to override externals install
if(DEFINED ENV{CIRCLE_MLIR_LOCALINST})
  set(CIRCLE_MLIR_LOCALINST $ENV{CIRCLE_MLIR_LOCALINST})
endif()

if(CIRCLE_MLIR_LOCALINST)
  message(STATUS "CIRCLE_MLIR_LOCALINST=${CIRCLE_MLIR_LOCALINST}")
endif()

set(RES_CIRCLE_SCHEMA "${CMAKE_SOURCE_DIR}/../res/CircleSchema")
if(NOT CIRCLE_MLIR_EXTERNALS)
  set(EXTERNALS_BIN_DIR "${CMAKE_BINARY_DIR}/externals")
else()
  set(EXTERNALS_BIN_DIR "${CMAKE_SOURCE_DIR}/${CIRCLE_MLIR_EXTERNALS}")
endif()

add_subdirectory(circle-mlir)
