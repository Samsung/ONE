cmake_minimum_required(VERSION 3.19)

set(APP_TARGET luci_micro_benchmark)

set(CMAKE_CXX_STANDARD 17)

################################################################################
#                             Configure environment                            #
################################################################################

set(CMAKE_FIND_LIBRARY_SUFFIXES .so .a)
set(CMAKE_FIND_LIBRARY_PREFIXES lib)

# Setup environment
find_package(PythonInterp 3 QUIET)
find_package(PythonLibs 3 QUIET)

if(NOT ${Python3_Interpreter_FOUND})
    message(FATAL_ERROR "Build MBED OS application Failed (Python3 is missing)")
endif()

if(${PYTHON_VERSION_MINOR} LESS 8)
    message(FATAL_ERROR "Build MBED OS application FAILED (You need to install Python version at least 3.8)")
endif()

# Create python virtual environment with mbed os
set(NNCC_OVERLAY_DIR "${CMAKE_BINARY_DIR}/overlay")
set(VIRTUALENV_OVERLAY_PATH_MBED_OS "${NNCC_OVERLAY_DIR}/venv_mdeb_os")

execute_process(COMMAND ${PYTHON_EXECUTABLE} -m venv ${VIRTUALENV_OVERLAY_PATH_MBED_OS})

# Create requirements.txt and install required pip packages
set(REQUIREMENTS_FILE "requirements.txt")
set(REQUIREMENTS_OVERLAY_PATH_MBED_OS "${VIRTUALENV_OVERLAY_PATH_MBED_OS}/${REQUIREMENTS_FILE}")

execute_process(COMMAND ${VIRTUALENV_OVERLAY_PATH_MBED_OS}/bin/python -m pip --default-timeout=1000 install --upgrade pip==20.2.1 setuptools==49.3.0)
execute_process(COMMAND ${CMAKE_COMMAND} -E echo "mbed-tools\nprettytable\nfuture\nintelhex" OUTPUT_FILE ${REQUIREMENTS_OVERLAY_PATH_MBED_OS})
execute_process(COMMAND ${VIRTUALENV_OVERLAY_PATH_MBED_OS}/bin/python -m pip --default-timeout=1000 install -r ${REQUIREMENTS_OVERLAY_PATH_MBED_OS} --upgrade)

# Replace system python with virtual env
# This is needed for mbed build
set (ENV{VIRTUAL_ENV} ${VIRTUALENV_OVERLAY_PATH_MBED_OS})
set (Python3_FIND_VIRTUALENV FIRST)
unset (Python3_EXECUTABLE)
find_package(PythonInterp 3 QUIET)

# download mbedos sources
set(NNAS_EXTERNALS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../externals)
set(MBED_OS_SOURCE_PATH "${NNAS_EXTERNALS_DIR}/mbed-os")
set(MBED_OS_LIB_PATH "${MBED_OS_SOURCE_PATH}.lib")
file(COPY mbed-os.lib DESTINATION ${NNAS_EXTERNALS_DIR})

# Download mbed os sources
if (NOT EXISTS "${MBED_OS_SOURCE_PATH}")
    execute_process(COMMAND ${VIRTUALENV_OVERLAY_PATH_MBED_OS}/bin/python -c "import sys; from mbed_tools.cli.main import cli; sys.exit(cli())" deploy -f ${NNAS_EXTERNALS_DIR})
    file(COPY GCC_ARM.cmake DESTINATION "${MBED_OS_SOURCE_PATH}/tools/cmake/toolchains/")
endif()

# MBED_CONFIG_PATH variable consumed in MBED scripts, do not delete it
set(MBED_CONFIG_PATH ${CMAKE_CURRENT_BINARY_DIR})
execute_process(COMMAND ${VIRTUALENV_OVERLAY_PATH_MBED_OS}/bin/python -c "import sys; from mbed_tools.cli.main import cli; sys.exit(cli())" configure -p ${NNAS_EXTERNALS_DIR} -m NUCLEO_F746ZG -t GCC_ARM -o ${MBED_CONFIG_PATH})
file(REMOVE "${MBED_OS_LIB_PATH}")

# Download compiler
function(_ArmNoneEabiToolchain_import)
    set(ARM_NONE_EABI_GCC_FILENAME "gcc-arm-none-eabi-9-2020-q2-update-x86_64-linux.tar.bz2")
    set(ARM_NONE_EABI_GCC_TOOLCHAIN_PATH "${NNAS_EXTERNALS_DIR}/gcc-arm-none-eabi-9-2020-q2-update" PARENT_SCOPE)
    set(ARM_NONE_EABI_URL "https://developer.arm.com/-/media/Files/downloads/gnu-rm/9-2020q2/${ARM_NONE_EABI_GCC_FILENAME}?revision=05382cca-1721-44e1-ae19-1e7c3dc96118&hash=CEB1348BF26C0285FD788E2424773FA304921735")

    if (NOT EXISTS "${ARM_NONE_EABI_GCC_TOOLCHAIN_PATH}")
        file(DOWNLOAD ${ARM_NONE_EABI_URL} "${ARM_NONE_EABI_GCC_TOOLCHAIN_PATH}.tar.bz2"
                EXPECTED_MD5 2b9eeccc33470f9d3cda26983b9d2dc6 STATUS result)
        if (NOT ${STATUS} EQUAL 0)
            message(FATAL_ERROR "Build MBED OS application Failed (Can not download compiler)")
        endif()
        execute_process(COMMAND tar -xjf "${ARM_NONE_EABI_GCC_TOOLCHAIN_PATH}.tar.bz2" -C "${NNAS_EXTERNALS_DIR}")
        # Q(binarman) Should add stamp here?
        # Network errors sometimes lead to inconsitensies of stamp and actual downloaded content, what do we need it for?
        file(REMOVE "${ARM_NONE_EABI_GCC_TOOLCHAIN_PATH}.tar.bz2")
    endif()
endfunction()

_ArmNoneEabiToolchain_import()

set(CMAKE_ASM_COMPILER "${ARM_NONE_EABI_GCC_TOOLCHAIN_PATH}/bin/arm-none-eabi-as")
set(CMAKE_C_COMPILER "${ARM_NONE_EABI_GCC_TOOLCHAIN_PATH}/bin/arm-none-eabi-gcc")
set(CMAKE_CXX_COMPILER "${ARM_NONE_EABI_GCC_TOOLCHAIN_PATH}/bin/arm-none-eabi-g++")
set(GCC_ELF2BIN "${ARM_NONE_EABI_GCC_TOOLCHAIN_PATH}/bin/arm-none-eabi-objcopy")

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)

################################################################################
#                                    Build                                     #
################################################################################


set(LUCI_MICRO ON)

set(FLATBUFFERS_SRC_DIR ${NNAS_EXTERNALS_DIR}/FLATBUFFERS/include CACHE INTERNAL "")

set(ONE_INFRA_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../infra CACHE INTERNAL "")
set(ONE_COMPILER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../ CACHE INTERNAL "")

set(TensorFlowSource_DIR ${NNAS_EXTERNALS_DIR}/TENSORFLOW-2.3.0 CACHE INTERNAL "")
set(TensorFlowGEMMLowpSource_DIR ${NNAS_EXTERNALS_DIR}/GEMMLOWP CACHE INTERNAL "")
set(TensorFlowEigenSource_DIR ${NNAS_EXTERNALS_DIR}/EIGEN CACHE INTERNAL "")
set(TensorFlowRuySource_DIR ${NNAS_EXTERNALS_DIR}/RUY CACHE INTERNAL "")


SET(GCC_COVERAGE_COMPILE_FLAGS "-fexceptions -Wno-register -frtti")
SET(GCC_COVERAGE_LINK_FLAGS    "")

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

include(${MBED_OS_SOURCE_PATH}/tools/cmake/app.cmake)

add_subdirectory(${MBED_OS_SOURCE_PATH} ${CMAKE_CURRENT_BINARY_DIR}/mbed-os)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/luci/lang ${CMAKE_CURRENT_BINARY_DIR}/luci/lang)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/loco ${CMAKE_CURRENT_BINARY_DIR}/loco)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/angkor ${CMAKE_CURRENT_BINARY_DIR}/angkor)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/oops ${CMAKE_CURRENT_BINARY_DIR}/oops)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/pepper-str ${CMAKE_CURRENT_BINARY_DIR}/pepper-str)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/logo ${CMAKE_CURRENT_BINARY_DIR}/logo)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/logo-core ${CMAKE_CURRENT_BINARY_DIR}/logo-core)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/locomotiv ${CMAKE_CURRENT_BINARY_DIR}/locomotiv)
add_subdirectory(${ONE_INFRA_SRC_DIR}/nncc ${CMAKE_CURRENT_BINARY_DIR}/nncc)
add_subdirectory(${ONE_COMPILER_SRC_DIR}/luci-interpreter ${CMAKE_CURRENT_BINARY_DIR}/luci-interpreter)

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
