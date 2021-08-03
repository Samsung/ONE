# BUILD ALGORITHM OVERVIEW
#
# if LUCI_MICRO_HOST_BUILD
#   build luci_interpreter_micro for x86
#
# if LUCI_MICRO_ARM_BUILD
#   if LUCI_MICRO_DOWNLOAD_COMPILER
#     download compiler
#     use downloaded compiler instead of default one
#   build luci_interpreter_micro for arm
#
# if LUCI_MICRO_MBEDOS_APP
#   download mdeb os
#   copy custom toolchain file to mbed os to build with downloaded compiler
#   build luci_interpreter along with mbed_os app
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

set_var_with_default(LUCI_MICRO_HOST_BUILD_ LUCI_MICRO_HOST_BUILD ON)
set_var_with_default(LUCI_MICRO_ARM_BUILD_ LUCI_MICRO_ARM_BUILD ON)
set_var_with_default(LUCI_MICRO_DOWNLOAD_ARM_COMPILER_ LUCI_MICRO_DOWNLOAD_ARM_COMPILER ON)
set_var_with_default(LUCI_MICRO_MBEDOS_APP_ LUCI_MICRO_MBEDOS_APP OFF)

# Need number of cpus to build in parallel
# This greatly reduces build time
include(ProcessorCount)
ProcessorCount(CPU_COUNT)

################################################################################
#                                 HOST build                                   #
################################################################################
if (${LUCI_MICRO_HOST_BUILD_})
  set(LUCI_INTERPRETER_STATIC ON)
  set(LUCI_STATIC ON)
  set(CUSTOM_LUCI_INTERPRETER_SUFFIX "_micro_host")
  add_subdirectory(../luci-interpreter ${CMAKE_CURRENT_BINARY_DIR}/luci-interpreter-micro-host)
endif()

################################################################################
#                                  ARM build                                   #
################################################################################

if (${LUCI_MICRO_ARM_BUILD_})
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
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DENABLE_TEST=OFF
  )

  file(MAKE_DIRECTORY "${MICRO_ARM_BUILD_DIR}")
  add_custom_command(
    OUTPUT "${MICRO_ARM_BUILD_DEPENDENCY}"
    COMMAND "${CMAKE_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/CrossCompile" ${CMAKE_ARM_OPTIONS}
    WORKING_DIRECTORY "${MICRO_ARM_BUILD_DIR}"
    VERBATIM
  )

  add_custom_target(luci_interpreter_micro_arm_prepare DEPENDS "${MICRO_ARM_BUILD_DEPENDENCY}")

  set(MICRO_ARM_BINARY "${MICRO_ARM_BUILD_DIR}/compiler/luci-interpreter/src/libluci_interpreter.a")

  add_custom_command(
    OUTPUT "${MICRO_ARM_BINARY}"
    COMMAND "${CMAKE_MAKE_PROGRAM}" luci_interpreter -j ${CPU_COUNT}
    WORKING_DIRECTORY "${MICRO_ARM_BUILD_DIR}"
    DEPENDS luci_interpreter_micro_arm_prepare
    VERBATIM
  )

  add_custom_target(luci_interpreter_micro_arm DEPENDS "${MICRO_ARM_BINARY}")
endif()

################################################################################
#                              MBED OS app build                               #
################################################################################

if(${LUCI_MICRO_MBEDOS_APP_})

  set(MICRO_ARM_BENCHMARK_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/arm_benchmark/")
  set(MICRO_ARM_BENCHMARK_BUILD_DEPENDENCY "${MICRO_ARM_BENCHMARK_BUILD_DIR}/CMakeCache.txt")

  set(CMAKE_ARM_BENCHMARK_OPTIONS
    -DLUCI_INTERPRETER_STATIC=ON
    -DLUCI_STATIC=ON
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DENABLE_TEST=OFF
  )

  file(MAKE_DIRECTORY "${MICRO_ARM_BENCHMARK_BUILD_DIR}")
  add_custom_command(
          OUTPUT "${MICRO_ARM_BENCHMARK_BUILD_DEPENDENCY}"
          COMMAND "${CMAKE_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/BenchmarkApp" ${CMAKE_ARM_BENCHMARK_OPTIONS}
          WORKING_DIRECTORY "${MICRO_ARM_BENCHMARK_BUILD_DIR}"
          VERBATIM
  )

  add_custom_target(luci_interpreter_micro_arm_benchmark_prepare DEPENDS "${MICRO_ARM_BENCHMARK_BUILD_DEPENDENCY}")

  set(MICRO_ARM_BENCHMARK "${MICRO_ARM_BENCHMARK_BUILD_DIR}/luci-micro/BenchmarkApp/luci_micro_benchmark")

  add_custom_command(
          OUTPUT "${MICRO_ARM_BENCHMARK}"
          COMMAND "${CMAKE_MAKE_PROGRAM}" luci_micro_arm_benchmark -j ${CPU_COUNT}
          WORKING_DIRECTORY "${MICRO_ARM_BUILD_DIR}"
          DEPENDS luci_interpreter_micro_arm_benchmark_prepare
          VERBATIM
  )

  add_custom_target(luci_interpreter_micro_arm_benchmark DEPENDS "${MICRO_ARM_BENCHMARK}")
endif()
