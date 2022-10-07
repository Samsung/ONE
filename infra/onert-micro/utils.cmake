set(NNAS_PROJECT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE
        INTERNAL "Where to find nnas top-level source directory"
        )

set(NNAS_EXTERNALS_DIR
        "${NNAS_PROJECT_SOURCE_DIR}/externals" CACHE
        INTERNAL "Where to download external dependencies"
        )
set(ONERT_MICRO_OVERLAY_DIR "${CMAKE_BINARY_DIR}/overlay" CACHE
        INTERNAL "Where locally built external dependencies are installed")

# Share package build script with runtime
set(EXT_OVERLAY_DIR ${ONERT_MICRO_OVERLAY_DIR})

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

###
### CMake configuration
###
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Type of build" FORCE)
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "Use '${CMAKE_BUILD_TYPE}' configuration")

# identify platform: HOST_PLATFORM, TARGET_PLATFORM and related
# note: this should be placed before flags and options setting
nnas_include(IdentifyPlatform)

# Configuration flags
include("${NNAS_PROJECT_SOURCE_DIR}/infra/onert-micro/cmake/CfgOptionFlags.cmake")

# apply compilation flags
# NOTE this should be after all option
include("${NNAS_PROJECT_SOURCE_DIR}/infra/onert-micro/cmake/ApplyCompileFlags.cmake")
