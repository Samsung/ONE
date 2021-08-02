# BUILD ALGORITHM OVERVIEW
#
# if not LUCI_MICRO
#   return
# build luci_interpreter_micro for x86
#
# if not LUCI_MICRO_ARM_BUILD
#   return
# if LUCI_MICRO_DOWNLOAD_COMPILER
#   download compiler
#   use downloaded compiler instead of default one
# build luci_interpreter_micro for arm
#
# if not LUCI_MICRO_MBEDOS_APP
#   return
# download mdeb os
# copy custom toolchain file to mbed os to build with downloaded compiler
# build mbed_os app
#

# Set internal variables, remove external from cache
# Do not use external variables itself, to prevent caching problems
macro(set_var_with_default DST_VAR SRC_VAR DEFAULT_VALUE)
  if (NOT DEFINED SRC_VAR)
    set(${DST_VAR} ${DEFAULT_VALUE})
  else()
    set(DST_VAR ${SRC_VAR})
  endif()
  unset(SRC_VAR CACHE) # prevent cmake from caching src variable
endmacro()

set_var_with_default(LUCI_MICRO_ LUCI_MICRO ON)
set_var_with_default(LUCI_MICRO_ARM_BUILD_ LUCI_MICRO_ARM_BUILD ON)
set_var_with_default(LUCI_MICRO_DOWNLOAD_ARM_COMPILER_ LUCI_MICRO_DOWNLOAD_ARM_COMPILER ON)
set_var_with_default(LUCI_MICRO_MBEDOS_APP_ LUCI_MICRO_MBEDOS_APP OFF)

# other general variables
set(ONE_COMPILER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.. CACHE INTERNAL "")
# Need number of cpus to build in parallel
# This greatly reduces build time
include(ProcessorCount)
ProcessorCount(CPU_COUNT)

# main sequence start
if (NOT ${LUCI_MICRO_})
  return()
endif()

set(MICRO_DEPENDENCIES_COMMON
        luci/lang
        loco
        angkor
        oops
        pepper-str
        logo
        logo-core
        locomotiv
        luci-interpreter
        )

################################################################################
#                                 HOST build                                   #
################################################################################

# build luci_micro for host architecture (probably x86)
# host build should be able to run tests, so include additional dependencies
set(MICRO_DEPENDENCIES_HOST
        foder
        pepper-csv2vec
        locop
        mio-circle
        hermes
        hermes-std
        arser
        mio-tflite
        safemain
        vconone
        tflite2circle
        luci
        ${MICRO_DEPENDENCIES_COMMON}
        )

set(MICRO_HOST_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/host/")
set(MICRO_HOST_BUILD_DEPENDENCY "${MICRO_HOST_BUILD_DIR}/CMakeCache.txt")

# TODO set PAl for this build
set(CMAKE_HOST_OPTIONS
        -DLUCI_INTERPRETER_STATIC=ON
        -DLUCI_STATIC=ON
        -DEXTERNALS_BUILD_THREADS=${CPU_COUNT})
file(MAKE_DIRECTORY "${MICRO_HOST_BUILD_DIR}")
add_custom_command(OUTPUT "${MICRO_HOST_BUILD_DEPENDENCY}"
        COMMAND "${CMAKE_COMMAND}" "${CMAKE_SOURCE_DIR}" ${CMAKE_HOST_OPTIONS} "-DBUILD_WHITELIST=${MICRO_DEPENDENCIES_HOST}"
        WORKING_DIRECTORY "${MICRO_HOST_BUILD_DIR}"
        DEPENDS luci_interpreter
        VERBATIM)

add_custom_target(luci_interpreter_micro_host_prepare DEPENDS "${MICRO_HOST_BUILD_DEPENDENCY}")

set(MICRO_HOST_BINARY "${MICRO_HOST_BUILD_DIR}/compiler/luci-interpreter/src/libluci_interpreter.a")

add_custom_command(OUTPUT "${MICRO_HOST_BINARY}"
        COMMAND "${CMAKE_MAKE_PROGRAM}" luci_interpreter -j ${CPU_COUNT}
        WORKING_DIRECTORY "${MICRO_HOST_BUILD_DIR}"
        DEPENDS luci_interpreter_micro_host_prepare)

add_custom_target(luci_interpreter_micro_host DEPENDS "${MICRO_HOST_BINARY}")

################################################################################
#                                  ARM build                                   #
################################################################################

if (NOT ${LUCI_MICRO_ARM_BUILD_})
  return()
endif()

# Configure toolchain
if (${LUCI_MICRO_DOWNLOAD_ARM_COMPILER_})
  # Download ARM none EABI toolchain
  nnas_find_package(ArmNoneEabiToolchain REQUIRED)
  set(ARM_C_COMPILER "${ArmNoneEabiToolchain_BINARY_DIR}/arm-none-eabi-gcc")
  set(ARM_ASM_COMPILER "${ArmNoneEabiToolchain_BINARY_DIR}/arm-none-eabi-gcc")
  set(ARM_CXX_COMPILER "${ArmNoneEabiToolchain_BINARY_DIR}/arm-none-eabi-g++")
  set(ARM_OBJCOPY "${ArmNoneEabiToolchain_BINARY_DIR}/arm-none-eabi-objcopy")
else()
  set(ARM_C_COMPILER "arm-none-eabi-gcc")
  set(ARM_ASM_COMPILER "arm-none-eabi-gcc")
  set(ARM_CXX_COMPILER "arm-none-eabi-g++")
  set(ARM_OBJCOPY "arm-none-eabi-objcopy")
endif()

set(MICRO_ARM_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/arm/")
set(MICRO_ARM_BUILD_DEPENDENCY "${MICRO_ARM_BUILD_DIR}/CMakeCache.txt")

# TODO set PAl for this build
set(CMAKE_ARM_OPTIONS
        -DLUCI_INTERPRETER_STATIC=ON
        -DLUCI_STATIC=ON
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/CrossCompile/Toolchain.cmake
        -DCPU_ARCH=arm
        -DC_COMPILER=${ARM_C_COMPILER}
        -DCXX_COMPILER=${ARM_CXX_COMPILER}
        -DASM_COMPILER=${ARM_ASM_COMPILER}
        -DOBJCOPY=${ARM_OBJCOPY})
file(MAKE_DIRECTORY "${MICRO_ARM_BUILD_DIR}")
add_custom_command(OUTPUT "${MICRO_ARM_BUILD_DEPENDENCY}"
        COMMAND "${CMAKE_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/CrossCompile" ${CMAKE_ARM_OPTIONS} "-DBUILD_COMPONENTS=${MICRO_DEPENDENCIES_COMMON}"
        WORKING_DIRECTORY "${MICRO_ARM_BUILD_DIR}"
        DEPENDS luci_interpreter
        VERBATIM)

add_custom_target(luci_interpreter_micro_arm_prepare DEPENDS "${MICRO_ARM_BUILD_DEPENDENCY}")

set(MICRO_ARM_BINARY "${MICRO_ARM_BUILD_DIR}/luci-interpreter/src/libluci_interpreter.a")

add_custom_command(OUTPUT "${MICRO_ARM_BINARY}"
        COMMAND "${CMAKE_MAKE_PROGRAM}" luci_interpreter -j ${CPU_COUNT}
        WORKING_DIRECTORY "${MICRO_ARM_BUILD_DIR}"
        DEPENDS luci_interpreter_micro_arm_prepare
        VERBATIM)

add_custom_target(luci_interpreter_micro_arm DEPENDS "${MICRO_ARM_BINARY}")

################################################################################
#                              MBED OS app build                               #
################################################################################

if (NOT ${LUCI_MICRO_MBEDOS_APP_})
  return()
endif()

# TODO download mbed os
# TODO build mbed os and link with benchmark

return()
# ***********************************************************************************************

set(MBED_PATH ${CMAKE_CURRENT_BINARY_DIR}/mbed-os CACHE INTERNAL "")
set(MBED_CONFIG_PATH ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")

# Configure toolchain
set(COMPILERS_DIR_PATH "${ARM_NONE_EABI_GCC_TOOLCHAIN_PATH}/bin")

set(CMAKE_C_COMPILER "${COMPILERS_DIR_PATH}/arm-none-eabi-gcc")
set(CMAKE_ASM_COMPILER "${COMPILERS_DIR_PATH}/arm-none-eabi-gcc")
set(CMAKE_CXX_COMPILER "${COMPILERS_DIR_PATH}/arm-none-eabi-g++")
set(GCC_ELF2BIN "${COMPILERS_DIR_PATH}/arm-none-eabi-objcopy")

set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)

set(APP_TARGET luci-micro)

# Install and deploy mbed os

# TODO do this in venv
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mbed-os.lib ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
execute_process(COMMAND pip3 install mbed-tools)

message("MBED DEPLOY")
#execute_process(COMMAND mbed-tools deploy -f ${CMAKE_CURRENT_BINARY_DIR})
message("MBED DEPLOYED")

# Download needed sources
execute_process(COMMAND git clone https://github.com/google/flatbuffers.git)
execute_process(COMMAND git clone https://gitlab.com/libeigen/eigen.git)
execute_process(COMMAND git clone https://github.com/google/gemmlowp.git)
execute_process(COMMAND git clone https://github.com/tensorflow/tensorflow.git)
execute_process(COMMAND git clone https://github.com/google/ruy.git)

# patch mbed os to comply with current cmake
#execute_process(COMMAND git apply ${CMAKE_CURRENT_BINARY_DIR}/mbed-os.patch WORKING_DIRECTORY ${MBED_PATH})

# copy custom toolchain file to toolchains
message("MBED CONFIGURE")
execute_process(COMMAND mbed-tools configure -p ${CMAKE_CURRENT_BINARY_DIR} -m NUCLEO_F746ZG -t GCC_ARM -o ${MBED_CONFIG_PATH})
message("MBED CONFIGURED")

set(FLATBUFFERS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers/include CACHE INTERNAL "")

set(ONE_COMPILER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.. CACHE INTERNAL "")
set(ONE_INFRA_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../infra CACHE INTERNAL "")

set(TensorFlowSource_DIR ${CMAKE_CURRENT_BINARY_DIR}/tensorflow CACHE INTERNAL "")
set(TensorFlowGEMMLowpSource_DIR ${CMAKE_CURRENT_BINARY_DIR}/gemmlowp CACHE INTERNAL "")
set(TensorFlowEigenSource_DIR ${CMAKE_CURRENT_BINARY_DIR}/eigen CACHE INTERNAL "")
set(TensorFlowRuySource_DIR ${CMAKE_CURRENT_BINARY_DIR}/ruy CACHE INTERNAL "")


set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_EXTENSIONS OFF)

SET(GCC_COVERAGE_COMPILE_FLAGS "-fexceptions -Wno-register -frtti")
SET(GCC_COVERAGE_LINK_FLAGS "")

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

message("MBED INCLUDE")

include(${MBED_PATH}/tools/cmake/app.cmake)

message("MBED INCLUDED")

add_subdirectory(${MBED_PATH} ${MBED_PATH}/build)
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

add_executable(${APP_TARGET} main.cpp)

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
