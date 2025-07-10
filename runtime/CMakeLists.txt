cmake_minimum_required(VERSION 3.16.3)

project(nnfw)

enable_testing()

# Library install path
include(GNUInstallDirs)
set(ONERT_INSTALL_APIDIR ${CMAKE_INSTALL_LIBDIR})
set(ONERT_INSTALL_COREDIR ${CMAKE_INSTALL_LIBDIR}/nnfw)
# ONERT core uses dlopen to use loader, backend, odc, and codegen.
# But asan build does not support dlopen with RPATH, so we need to install all libraries in same directory with core.
if (ASAN_BUILD)
  set(ONERT_INSTALL_LOADERDIR ${ONERT_INSTALL_COREDIR})
  set(ONERT_INSTALL_BACKENDDIR ${ONERT_INSTALL_COREDIR})
  set(ONERT_INSTALL_ODCDIR ${ONERT_INSTALL_COREDIR})
  set(ONERT_INSTALL_CODEGENDIR ${ONERT_INSTALL_COREDIR})
else (ASAN_BUILD)
  set(ONERT_INSTALL_LOADERDIR ${ONERT_INSTALL_COREDIR}/loader)
  set(ONERT_INSTALL_BACKENDDIR ${ONERT_INSTALL_COREDIR}/backend)
  set(ONERT_INSTALL_ODCDIR ${ONERT_INSTALL_COREDIR}/odc)
  set(ONERT_INSTALL_CODEGENDIR ${ONERT_INSTALL_COREDIR}/codegen)
endif(ASAN_BUILD)
# Decide rpath for runtime library here because install path is decided above
# API rpath: find core
# core rpath: find plugins - loader, backend, odc, codegen
# plugin rpath: find core
set(ONERT_RPATH_API "$ORIGIN:$ORIGIN/nnfw")
set(ONERT_RPATH_CORE "$ORIGIN:$ORIGIN/loader:$ORIGIN/backend:$ORIGIN/odc:$ORIGIN/codegen")
set(ONERT_RPATH_PLUGIN "$ORIGIN:$ORIGIN/..")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/:$ORIGIN/")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

### CMAKE_BUILD_TYPE_LC: Build type lower case
string(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_LC)

### Set source directories to use in global scope macro & function
set(ONE_PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(ONERT_PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(NNFW_EXTERNALS_DIR
  "${ONERT_PROJECT_SOURCE_DIR}/externals" CACHE
  INTERNAL "Where to download external dependencies"
)
set(NNFW_OVERLAY_DIR "${CMAKE_BINARY_DIR}/overlay" CACHE
    INTERNAL "Where locally built external dependencies are installed")

# Share package build script with compiler
set(EXT_OVERLAY_DIR ${NNFW_OVERLAY_DIR})

# This allows find_package configurations, find_library, and find_path to access installed inside overlay
# EXT_OVERLAY_DIR is higher priority than ROOTFS_DIR on cross build
list(INSERT CMAKE_FIND_ROOT_PATH 0 "${EXT_OVERLAY_DIR}")

macro(nnfw_include PREFIX)
  include("${ONERT_PROJECT_SOURCE_DIR}/infra/cmake/modules/${PREFIX}.cmake")
endmacro(nnfw_include)

# Runtime 'find_package()' wrapper to find in cmake/packages folder
#
# Example:
#  nnfw_find_package(Eigen): Load settings from 'EigenConfig.cmake' file
#    - this may drop warnings like "-- Could NOT find Eigen (missing: Eigen_DIR)
#  nnfw_find_package(Eigen QUIET): Load settings silently, without warnings
#  nnfw_find_package(Eigen REQUIRED): Load settings but stop with error when failed
macro(nnfw_find_package PREFIX)
  find_package(${PREFIX} CONFIG NO_DEFAULT_PATH
    PATHS ${ONERT_PROJECT_SOURCE_DIR}/infra/cmake/packages
    ${ARGN}
  )
endmacro(nnfw_find_package)

# C++17 feature requires 9.1 or later for stable usage
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
  message(FATAL "Runtime build requires GNU Compiler version 9.1 or later.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# identify platform: HOST_PLATFORM, TARGET_PLATFORM and related
# note: this should be placed before flags and options setting
nnfw_include(IdentifyPlatform)

# Configuration flags
include("${CMAKE_CURRENT_LIST_DIR}/infra/cmake/CfgOptionFlags.cmake")
# and besides CfgOptionFlags.cmake that can be given outside
#   COVERAGE_BUILD: build boolean flag that enables converage test
#   ROOTFS_DIR: rootfs path for cross building
#   TARGET_ARCH: target architecture string for cross building
#   TARGET_OS: target os string for cross building

# apply compilation flags
# NOTE This should be placed after cmake/CfgOptionFlags.cmake files include
#      because compile flag setting can be decided using option (ex. ENABLE_COVERAGE)
include("${CMAKE_CURRENT_LIST_DIR}/infra/cmake/ApplyCompileFlags.cmake")

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

nnfw_include(AddSubdirectories)

# Add arser for test driver
# TODO: Better way to handle this
if(ENABLE_TEST)
  add_library(arser INTERFACE)
  target_include_directories(arser INTERFACE ${ONE_PROJECT_SOURCE_DIR}/compiler/arser/include/)
endif(ENABLE_TEST)

add_subdirectory(3rdparty)
add_subdirectory(compute)
add_subdirectory(contrib)
add_subdirectory(libs)
add_subdirectory(onert)
add_subdirectory(tests)
add_subdirectory(tools)
