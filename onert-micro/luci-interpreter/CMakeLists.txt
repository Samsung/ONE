set(LUCI_INTERPRETER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(LUCI_INTERPRETER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(LUCI_INTERPRETER_PAL_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pal/common")
if (NOT LUCI_INTERPRETER_PAL_DIR)
    set(LUCI_INTERPRETER_PAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pal/mcu")
endif()

if (NOT LUCI_INTERPRETER_KERNELS_BUILD_LIST)
    set(KERNEL_REGISTER_FILE "${LUCI_INTERPRETER_PAL_DIR}/KernelsToBuild.lst")
else()
    set(KERNEL_REGISTER_FILE ${LUCI_INTERPRETER_KERNELS_BUILD_LIST})
endif()

if (NOT DEFINED CUSTOM_LUCI_INTERPRETER_SUFFIX)
    set(LUCI_INTERPRETER_SUFFIX "")
else()
    set(LUCI_INTERPRETER_SUFFIX ${CUSTOM_LUCI_INTERPRETER_SUFFIX})
endif()

if (DIS_QUANT)
    add_definitions(-DDIS_QUANT)
endif()

if (DIS_FLOAT)
    add_definitions(-DDIS_FLOAT)
endif()

add_compile_options(-fno-exceptions)
#add_compile_options(-Os)

# AFAIK, this will enable leak sanitizer, too
if(ENABLE_SANITIZER)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif(ENABLE_SANITIZER)

add_subdirectory(src)
