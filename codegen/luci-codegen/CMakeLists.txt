if(NOT DEFINED Halide_DIR)
  nnas_find_package(Halide REQUIRED)
else(NOT DEFINED Halide_DIR)
  find_package(Halide REQUIRED)
endif(NOT DEFINED Halide_DIR)

if (NOT Halide_FOUND)
    message(STATUS "Skipping luci-codegen: Halide not found, see README.md for instructions")
    return()
endif ()

set(LUCI_CODEGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(LUCI_CODEGEN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(LUCI_CODEGEN_SUPPORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/support")

add_subdirectory(filesystem)
add_subdirectory(src)
