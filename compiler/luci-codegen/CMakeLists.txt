find_package(Halide REQUIRED)

if (NOT Halide_FOUND)
    message(STATUS "Skipping luci-codegen: Halide not found, see README.md for instructions")
    return()
endif ()

nnas_find_package(Boost REQUIRED COMPONENTS filesystem)

if (NOT Boost_FOUND)
    message(STATUS "Skipping luci-codegen: Boost not found")
    return()
endif()

set(LUCI_CODEGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(LUCI_CODEGEN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(LUCI_CODEGEN_SUPPORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/support")

add_subdirectory(src)
