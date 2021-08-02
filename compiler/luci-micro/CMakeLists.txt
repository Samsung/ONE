# OVERVIEW
#
# build luci_micro for x86
# if MICRO_INTERPRETER_ARM_BUILD
#   if MICRO_INTERPRETER_DOWNLOAD_COMPILER
#     download compiler
#     set downloaded compiler as target
#
#   set toolchain cmake
#   build_arm_luci_micro
#   if MICRO_INTERPRETER_MBEDOS_APP
#     download mdebos
#     copy toolchain file to mbedos
#     build mbed_os app
#

set(LUCI_MICRO ON CACHE BOOL "LUCI_MICRO BUILD FLAG")
set(MBED_PATH ${CMAKE_CURRENT_SOURCE_DIR}/mbed-os CACHE INTERNAL "")
set(MBED_CONFIG_PATH ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")

#set(MBED_CONFIG_PATH ${CMAKE_CURRENT_SOURCE_DIR}/.mbedbuild CACHE INTERNAL "")

# Download ARM none EABI toolchain
set(ARM_NONE_EABI_GCC_FILENAME "gcc-arm-none-eabi-9-2020-q2-update-x86_64-linux.tar.bz2")
set(ARM_NONE_EABI_GCC_TOOLCHAIN_PATH "${CMAKE_CURRENT_SOURCE_DIR}/gcc-arm-none-eabi-9-2020-q2-update")
set(ARM_NONE_EABI_URL "https://developer.arm.com/-/media/Files/downloads/gnu-rm/9-2020q2/${ARM_NONE_EABI_GCC_FILENAME}?revision=05382cca-1721-44e1-ae19-1e7c3dc96118&hash=CEB1348BF26C0285FD788E2424773FA304921735")
if (NOT EXISTS "${ARM_NONE_EABI_GCC_TOOLCHAIN_PATH}.tar.bz2")
    file(DOWNLOAD ${ARM_NONE_EABI_URL} "${ARM_NONE_EABI_GCC_TOOLCHAIN_PATH}.tar.bz2"
            EXPECTED_MD5 2b9eeccc33470f9d3cda26983b9d2dc6)

    execute_process(COMMAND tar -xjf "${ARM_NONE_EABI_GCC_TOOLCHAIN_PATH}.tar.bz2")
endif ()

set(COMPILERS_DIR_PATH "${ARM_NONE_EABI_GCC_TOOLCHAIN_PATH}/bin")

set(CMAKE_C_COMPILER "${COMPILERS_DIR_PATH}/arm-none-eabi-gcc")
set(CMAKE_ASM_COMPILER "${COMPILERS_DIR_PATH}/arm-none-eabi-gcc")
set(CMAKE_CXX_COMPILER "${COMPILERS_DIR_PATH}/arm-none-eabi-g++")
set(GCC_ELF2BIN "${COMPILERS_DIR_PATH}/arm-none-eabi-objcopy")

set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)

set(APP_TARGET luci-micro)

execute_process(COMMAND pip3 install mbed-tools)
execute_process(COMMAND mbed-tools deploy -f ${CMAKE_CURRENT_SOURCE_DIR})
execute_process(COMMAND python3.9 ${CMAKE_CURRENT_SOURCE_DIR}/import_lib.py)
execute_process(COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/mbed-os.patch WORKING_DIRECTORY ${MBED_PATH})
execute_process(COMMAND mbed-tools configure -m NUCLEO_F746ZG -t GCC_ARM -o ${MBED_CONFIG_PATH})

set(FLATBUFFERS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers/include CACHE INTERNAL "")

set(ONE_COMPILER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.. CACHE INTERNAL "")
set(ONE_INFRA_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../infra CACHE INTERNAL "")

set(TensorFlowSource_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow CACHE INTERNAL "")
set(TensorFlowGEMMLowpSource_DIR ${CMAKE_CURRENT_SOURCE_DIR}/gemmlowp CACHE INTERNAL "")
set(TensorFlowEigenSource_DIR ${CMAKE_CURRENT_SOURCE_DIR}/eigen CACHE INTERNAL "")
set(TensorFlowRuySource_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ruy CACHE INTERNAL "")


set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_EXTENSIONS OFF)

SET(GCC_COVERAGE_COMPILE_FLAGS "-fexceptions -Wno-register -frtti")
SET(GCC_COVERAGE_LINK_FLAGS    "")

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

include(${MBED_PATH}/tools/cmake/app.cmake)

add_subdirectory(${MBED_PATH})
add_subdirectory(${ONE_COMPILER_SRC_DIR}/luci/lang ${CMAKE_CURRENT_SOURCE_DIR}/luci/lang)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/loco ${CMAKE_CURRENT_SOURCE_DIR}/loco)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/angkor ${CMAKE_CURRENT_SOURCE_DIR}/angkor)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/oops ${CMAKE_CURRENT_SOURCE_DIR}/oops)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/pepper-str ${CMAKE_CURRENT_SOURCE_DIR}/pepper-str)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/logo ${CMAKE_CURRENT_SOURCE_DIR}/logo)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/logo-core ${CMAKE_CURRENT_SOURCE_DIR}/logo-core)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/locomotiv ${CMAKE_CURRENT_SOURCE_DIR}/locomotiv)
add_subdirectory(${ONE_INFRA_SRC_DIR}/nncc ${CMAKE_CURRENT_SOURCE_DIR}/nncc)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/luci-interpreter ${CMAKE_CURRENT_SOURCE_DIR}/luci-interpreter)

add_executable(${APP_TARGET}
        main.cpp
        )

project(${APP_TARGET})

target_link_libraries(${APP_TARGET} PRIVATE mbed-os)
target_link_libraries(${APP_TARGET} PRIVATE luci_lang)
target_link_libraries(${APP_TARGET} PRIVATE loco)
target_link_libraries(${APP_TARGET} PRIVATE angkor)
target_link_libraries(${APP_TARGET} PRIVATE oops)
target_link_libraries(${APP_TARGET} PRIVATE pepper_str)
target_link_libraries(${APP_TARGET} PRIVATE logo)
target_link_libraries(${APP_TARGET} PRIVATE logo_core)
target_link_libraries(${APP_TARGET} PRIVATE locomotiv)
target_link_libraries(${APP_TARGET} PRIVATE luci_interpreter)

mbed_set_post_build(${APP_TARGET})

option(VERBOSE_BUILD "Have a verbose build process")
if(VERBOSE_BUILD)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif()
