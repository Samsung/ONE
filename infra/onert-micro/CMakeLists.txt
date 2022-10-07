cmake_minimum_required(VERSION 3.15)

project(onert-micro)

enable_testing()

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(utils.cmake)

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

###
### Target
###
add_library(onert_micro_common INTERFACE)
if(ENABLE_STRICT_BUILD)
    target_compile_options(onert_micro_common INTERFACE -Werror -Wall -Wextra -Wno-reorder)
endif(ENABLE_STRICT_BUILD)

add_library(onert_micro_coverage INTERFACE)
if(ENABLE_COVERAGE)
    target_compile_options(onert_micro_coverage INTERFACE -g -O0 -fprofile-arcs -ftest-coverage)
    target_link_libraries(onert_micro_coverage INTERFACE gcov)
endif(ENABLE_COVERAGE)

add_subdirectory("${NNAS_PROJECT_SOURCE_DIR}/onert-micro" "${CMAKE_BINARY_DIR}/onert-micro")
