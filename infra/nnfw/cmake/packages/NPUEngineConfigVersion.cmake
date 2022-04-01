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

set(NPU_ENGINE_PREFIX "/usr" CACHE PATH "Where to find NPU engine header and library")

if(NOT PACKAGE_FIND_VERSION)
  message(FATAL_ERROR "Please pass version requirement to use NPU Engine dependency")
endif()

# Find the header & lib from NPU_ENGINE_PREFIX
find_library(NPUEngine_LIB
  NAMES npu-engine
  HINTS "${NPU_ENGINE_PREFIX}/lib"
)
find_path(NPUEngine_INCLUDE_DIR
  NAMES libnpuhost.h
  HINTS "${NPU_ENGINE_PREFIX}/include/npu-engine"
)

if(NOT NPUEngine_INCLUDE_DIR OR NOT NPUEngine_LIB)
  set(PACKAGE_VERSION_EXACT FALSE)
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
  set(PACKAGE_VERSION_UNSUITABLE TRUE)
  return()
endif(NOT NPUEngine_INCLUDE_DIR OR NOT NPUEngine_LIB)

# TODO Assert NPU_ENGINE_PREFIX is directory

# TODO Can we run this only once per configure?
try_run(MAJOR_VER MAJOR_COMPILABLE "${CMAKE_BINARY_DIR}/NPUEngineConfigVersion.major"
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/NPUEngineConfigVersion.major.cpp"
  CMAKE_FLAGS
  "-DINCLUDE_DIRECTORIES=${NPUEngine_INCLUDE_DIR}"
  "-DLINK_LIBRARIES=${NPUEngine_LIB}"
)

if(NOT MAJOR_COMPILABLE)
  # This means VERSION < 2.2.7
  # `getVersion` API introduced from NPU Engine 2.2.7
  if(PACKAGE_FIND_VERSION VERSION_GREATER_EQUAL 2.2.7)
    set(PACKAGE_VERSION_EXACT FALSE)
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
    set(PACKAGE_VERSION_UNSUITABLE TRUE)
    return()
  else()
    # TODO How to support this case?
    message(FATAL_ERROR "NPU Engine version is too low (< 2.2.7)")
  endif()
endif(NOT MAJOR_COMPILABLE)

try_run(MINOR_VER MINOR_COMPILABLE "${CMAKE_BINARY_DIR}/NPUEngineConfigVersion.minor"
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/NPUEngineConfigVersion.minor.cpp"
  CMAKE_FLAGS
  "-DINCLUDE_DIRECTORIES=${NPUEngine_INCLUDE_DIR}"
  "-DLINK_LIBRARIES=${NPUEngine_LIB}"
)

try_run(EXTRA_VER EXTRA_COMPILABLE "${CMAKE_BINARY_DIR}/NPUEngineConfigVersion.extra"
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/NPUEngineConfigVersion.extra.cpp"
  CMAKE_FLAGS
  "-DINCLUDE_DIRECTORIES=${NPUEngine_INCLUDE_DIR}"
  "-DLINK_LIBRARIES=${NPUEngine_LIB}"
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

if(PACKAGE_VERSION VERSION_EQUAL PACKAGE_FIND_VERSION)
  set(PACKAGE_VERSION_EXACT TRUE)
else()
  set(PACKAGE_VERSION_EXACT FALSE)
endif()

# Assume NPU Engine is backward compatible
if(PACKAGE_VERSION VERSION_GREATER_EQUAL PACKAGE_FIND_VERSION)
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
else()
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
endif()

set(PACKAGE_VERSION_UNSUITABLE FALSE)
