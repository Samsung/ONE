cmake_minimum_required(VERSION 3.16.3)

project(nnfw)

enable_testing()

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/../lib/nnfw:$ORIGIN/")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

### CMAKE_BUILD_TYPE_LC: Build type lower case
string(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_LC)

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

# C++14 feature requires 5 or later
# Using std::unordered_map shows build fail under 6.2
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.2)
  message(FATAL "Runtime build requires GNU Compiler version 6.2 or later.")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# identify platform: HOST_PLATFORM, TARGET_PLATFORM and related
# note: this should be placed before flags and options setting
nnas_include(IdentifyPlatform)

# Configuration flags
include("cmake/CfgOptionFlags.cmake")
# and besides CfgOptionFlags.cmake that can be given outside
#   COVERAGE_BUILD: build boolean flag that enables converage test
#   ROOTFS_DIR: rootfs path for cross building
#   TARGET_ARCH: target architecture string for cross building
#   TARGET_OS: target os string for cross building

# apply compilation flags
# NOTE This should be placed after cmake/CfgOptionFlags.cmake files include
#      because compile flag setting can be decided using option (ex. ENABLE_COVERAGE)
include("cmake/ApplyCompileFlags.cmake")

nnfw_find_package(GTest QUIET)

option(ENABLE_TEST "Build Tests using Google Test" ${GTest_FOUND})

if(${ENABLE_TEST} AND NOT ${GTest_FOUND})
  message(FATAL_ERROR "Google Test is required to enable test")
endif(${ENABLE_TEST} AND NOT ${GTest_FOUND})

if(${ENABLE_COVERAGE} AND NOT ${ENABLE_TEST})
  message(FATAL_ERROR "Test should be enabled to measure test coverage")
endif(${ENABLE_COVERAGE} AND NOT ${ENABLE_TEST})

add_library(nnfw_common INTERFACE)
if(ENABLE_STRICT_BUILD)
  target_compile_options(nnfw_common INTERFACE -Werror -Wall -Wextra)
endif(ENABLE_STRICT_BUILD)

macro(nnfw_strict_build TARGET)
  if(ENABLE_STRICT_BUILD)
    target_compile_options(${TARGET} PRIVATE -Werror -Wall -Wextra)
  endif(ENABLE_STRICT_BUILD)
endmacro(nnfw_strict_build)

# TODO Replace using default build option setting in cmake/buildtool/config/config_linux.cmake
#      to link nnfw_coverage on each module which want to check coverage
add_library(nnfw_coverage INTERFACE)
if(ENABLE_COVERAGE)
  target_compile_options(nnfw_coverage INTERFACE -fprofile-arcs -ftest-coverage)
  target_link_libraries(nnfw_coverage INTERFACE gcov)
endif(ENABLE_COVERAGE)

nnas_include(AddSubdirectories)

add_subdirectory(${NNAS_PROJECT_SOURCE_DIR}/compute compute)
add_subdirectory(${NNAS_PROJECT_SOURCE_DIR}/runtime runtime)
add_subdirectory(${NNAS_PROJECT_SOURCE_DIR}/tests tests)
add_subdirectory(${NNAS_PROJECT_SOURCE_DIR}/tools tools)
