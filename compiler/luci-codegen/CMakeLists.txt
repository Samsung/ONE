find_package(Halide REQUIRED)

if (NOT Halide_FOUND)
    message(STATUS "Skipping luci-codegen: Halide not found, see README.md for instructions")
    return()
endif ()

set(LUCI_CODEGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(LUCI_CODEGEN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_subdirectory(src)
