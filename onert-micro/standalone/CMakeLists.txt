cmake_minimum_required(VERSION 3.15)
project(luci_interpreter_micro_standalone)

include(${NNAS_ROOT}/infra/onert-micro/utils.cmake)

if (NOT ${NOT_BUILD_EXTERNALS})
    nnas_find_package(FlatBuffersSource EXACT 2.0 QUIET)
    include_directories(${FlatBuffersSource_DIR}/include)
else()
    include_directories(${FlatBuffersSource_DIR})
endif()

# TODO: fix luci/plan for new luci-micro without luci/IR
add_subdirectory(${NNAS_PROJECT_SOURCE_DIR}/onert-micro/luci-interpreter ${CMAKE_CURRENT_BINARY_DIR}/luci-interpreter)
