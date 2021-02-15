find_package(Halide REQUIRED)

if (NOT Halide_FOUND)
    message(STATUS "Skipping luci-codegen: Halide not found")
    return()
endif ()

find_package(LLVM REQUIRED)

if (NOT LLVM_FOUND)
    message(STATUS "Skipping luci-codegen: LLVM not found")
    return()
endif ()

llvm_map_components_to_libnames(llvm_libs support core)

set(LUCI_CODEGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(LUCI_CODEGEN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_subdirectory(src)
