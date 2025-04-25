# This script need to set:
#
# VARIABLE                   | description
# ---                        | ---
# PACKAGE_VERSION            | full provided version string
# PACKAGE_VERSION_EXACT      | true if version is exact match
# PACKAGE_VERSION_COMPATIBLE | true if version is compatible
# PACKAGE_VERSION_UNSUITABLE | true if unsuitable as any version
#
# Reference: https://cmake.org/cmake/help/v3.10/command/find_package.html

set(TRIX_ENGINE_PREFIX "/usr" CACHE PATH "Where to find TRIX engine header and library")

if(NOT PACKAGE_FIND_VERSION)
  message(FATAL_ERROR "Please pass version requirement to use TRIX Engine dependency")
endif()

# Find the header & lib from TRIX_ENGINE_PREFIX
find_library(TRIXEngine_LIB
  NAMES npu-engine
  HINTS "${TRIX_ENGINE_PREFIX}/lib"
)
find_path(TRIXEngine_INCLUDE_DIR
  NAMES libnpuhost.h
  HINTS "${TRIX_ENGINE_PREFIX}/include/npu-engine"
)

# npubinfmt.h is in different path on version >=2.6.0
find_path(NPUBINFMT_INCLUDE_DIR
  NAMES npubinfmt.h
  HINTS "${TRIX_ENGINE_PREFIX}/include/npubin-fmt"
        "${TRIX_ENGINE_PREFIX}/include/npu-engine"
)

if(NOT TRIXEngine_INCLUDE_DIR OR NOT NPUBINFMT_INCLUDE_DIR OR NOT TRIXEngine_LIB)
  message(STATUS "Fail to check TRIXEngine version")
  set(PACKAGE_VERSION_EXACT FALSE)
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
  set(PACKAGE_VERSION_UNSUITABLE TRUE)
  return()
endif(NOT TRIXEngine_INCLUDE_DIR OR NOT NPUBINFMT_INCLUDE_DIR OR NOT TRIXEngine_LIB)

set(TRYRUN_COMPILE_DEFINITIONS "-I${TRIXEngine_INCLUDE_DIR} -I${NPUBINFMT_INCLUDE_DIR}")

# TODO Assert TRIX_ENGINE_PREFIX is directory

# TODO Can we run this only once per configure?
try_run(MAJOR_VER MAJOR_COMPILABLE "${CMAKE_BINARY_DIR}/TRIXEngineConfigVersion.major"
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/TRIXEngineConfigVersion.major.cpp"
  COMPILE_DEFINITIONS "${TRYRUN_COMPILE_DEFINITIONS}"
  LINK_LIBRARIES ${TRIXEngine_LIB}
)

if(NOT MAJOR_COMPILABLE)
  # This means VERSION < 2.2.7
  # `getVersion` API introduced from TRIX Engine 2.2.7
  if(PACKAGE_FIND_VERSION VERSION_GREATER_EQUAL 2.2.7)
    message(STATUS "Fail to build TRIXEngine version checker")
    set(PACKAGE_VERSION_EXACT FALSE)
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
    set(PACKAGE_VERSION_UNSUITABLE TRUE)
    return()
  else()
    # TODO How to support this case?
    message(FATAL_ERROR "TRIX Engine version is too low (< 2.2.7)")
  endif()
endif(NOT MAJOR_COMPILABLE)

try_run(MINOR_VER MINOR_COMPILABLE "${CMAKE_BINARY_DIR}/TRIXEngineConfigVersion.minor"
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/TRIXEngineConfigVersion.minor.cpp"
  COMPILE_DEFINITIONS "${TRYRUN_COMPILE_DEFINITIONS}"
  LINK_LIBRARIES ${TRIXEngine_LIB}
)

try_run(EXTRA_VER EXTRA_COMPILABLE "${CMAKE_BINARY_DIR}/TRIXEngineConfigVersion.extra"
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/TRIXEngineConfigVersion.extra.cpp"
  COMPILE_DEFINITIONS "${TRYRUN_COMPILE_DEFINITIONS}"
  LINK_LIBRARIES ${TRIXEngine_LIB}
)

macro(assert)
  # if(NOT ${ARGV}) makes error when ARGV starts with 'NOT'
  if(${ARGV})
    # Do nothing
  else(${ARGV})
    message(FATAL_ERROR "Internal error ${ARGV}")
  endif(${ARGV})
endmacro(assert)

assert(MAJOR_COMPILABLE)
assert(MINOR_COMPILABLE)
assert(EXTRA_COMPILABLE)
assert(NOT MAJOR_VER STREQUAL FAILED_TO_RUN)
assert(NOT MINOR_VER STREQUAL FAILED_TO_RUN)
assert(NOT EXTRA_VER STREQUAL FAILED_TO_RUN)

set(PACKAGE_VERSION ${MAJOR_VER}.${MINOR_VER}.${EXTRA_VER})
message(STATUS "Found TRIXEngine: ${PACKAGE_VERSION}")

if(PACKAGE_VERSION VERSION_EQUAL PACKAGE_FIND_VERSION)
  message("Found TRIXEngine: ${PACKAGE_VERSION} exact match")
  set(PACKAGE_VERSION_EXACT TRUE)
else()
  set(PACKAGE_VERSION_EXACT FALSE)
endif()

# Assume TRIX Engine is backward compatible
if(PACKAGE_VERSION VERSION_GREATER_EQUAL PACKAGE_FIND_VERSION)
  message("Found TRIXEngine: ${PACKAGE_VERSION} compatible with ${PACKAGE_FIND_VERSION}")
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
else()
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
endif()

set(PACKAGE_VERSION_UNSUITABLE FALSE)
