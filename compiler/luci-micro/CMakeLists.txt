set(ARM_C_COMPILER "arm-none-eabi-gcc")
set(ARM_ASM_COMPILER "arm-none-eabi-gcc")
set(ARM_CXX_COMPILER "arm-none-eabi-g++")
set(ARM_OBJCOPY "arm-none-eabi-objcopy")

find_program(ARM_C_COMPILER_PATH ${ARM_C_COMPILER})

if (NOT ARM_C_COMPILER_PATH)
    message(WARNING "ARM compiler is NOT FOUND, skipping luci-micro build")
    return()
endif ()
set(ONE_COMPILER_SRC_DIR "${NNAS_PROJECT_SOURCE_DIR}/compiler")

set(CMAKE_ARM_OPTIONS
        -DNNCC_LIBRARY_NO_PIC=ON
        -DLUCI_INTERPRETER_STATIC=ON
        -DLUCI_STATIC=ON
        -DBUILD_CMSIS_NN_FUNCTIONS=ON
        -DTARGET_CPU=cortex-m7
        "-DCMAKE_TOOLCHAIN_FILE=${NNAS_PROJECT_SOURCE_DIR}/infra/nncc/cmake/buildtool/config/arm-none-eabi-gcc.cmake"
        "-DLUCI_INTERPRETER_PAL_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../luci-interpreter/pal/cmsisnn"
        "-DNNAS_PROJECT_SOURCE_DIR=${NNAS_PROJECT_SOURCE_DIR}"
        "-DNNAS_EXTERNALS_DIR=${NNAS_EXTERNALS_DIR}"
        -DC_COMPILER=${ARM_C_COMPILER}
        -DCXX_COMPILER=${ARM_CXX_COMPILER}
        -DASM_COMPILER=${ARM_ASM_COMPILER}
        -DOBJCOPY=${ARM_OBJCOPY}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DENABLE_TEST=OFF
        -DBUILD_GTEST=OFF
        "-DNNAS_ROOT=${NNAS_PROJECT_SOURCE_DIR}"
        -DENABLE_STRICT_BUILD=OFF
        -DHERMES_NO_PIC=ON
        -DHERMES_STD_NO_PIC=ON
        -DANGKOR_NO_PIC=ON
        -DLUCI_INTERPRETER_KERNELS_NO_PIC=ON
        -DLUCI_INTERPRETER_CORE_NO_PIC=ON
        -DLUCI_INTERPRETER_LOADER_NO_PIC=ON
        -DLOCOMOTIV_NO_PIC=ON
        -DLOGO_NO_PIC=ON
        -DLOGO_CORE_NO_PIC=ON
        -DLOCOP_NO_PIC=ON
        -DPP_NO_PIC=ON
        -DPEPPER_STRCAST_NO_PIC=ON
        )

set(MICRO_ARM_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/standalone_arm")
file(MAKE_DIRECTORY "${MICRO_ARM_BUILD_DIR}")

set(MICRO_ARM_BUILD_DEPENDENCY "${MICRO_ARM_BUILD_DIR}/CMakeCache.txt")

add_custom_command(
        OUTPUT "${MICRO_ARM_BUILD_DEPENDENCY}"
        COMMAND "${CMAKE_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/standalone" ${CMAKE_ARM_OPTIONS}
        WORKING_DIRECTORY "${MICRO_ARM_BUILD_DIR}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/standalone/CMakeLists.txt"
        VERBATIM
)

add_custom_target(luci_interpreter_micro_arm_cmake DEPENDS "${MICRO_ARM_BUILD_DEPENDENCY}")

set(MICRO_ARM_BINARY "${MICRO_ARM_BUILD_DIR}/compiler/luci-interpreter/src/libluci_interpreter.a")

add_custom_command(
        OUTPUT "${MICRO_ARM_BINARY}"
        COMMAND "${CMAKE_MAKE_PROGRAM}" luci_interpreter -j ${CPU_COUNT}
        WORKING_DIRECTORY "${MICRO_ARM_BUILD_DIR}"
        DEPENDS luci_interpreter_micro_arm_cmake
        VERBATIM
)
add_custom_target(luci_interpreter_micro_arm DEPENDS "${MICRO_ARM_BINARY}")

set(MICRO_LUCI_IMPORT_ARM_BINARY "${MICRO_ARM_BUILD_DIR}/compiler/luci/import/libluci_import.a")

add_custom_command(
        OUTPUT "${MICRO_LUCI_IMPORT_ARM_BINARY}"
        COMMAND "${CMAKE_MAKE_PROGRAM}" luci_import -j ${CPU_COUNT}
        WORKING_DIRECTORY "${MICRO_ARM_BUILD_DIR}"
        DEPENDS luci_interpreter_micro_arm_cmake
        VERBATIM
)

add_custom_target(luci_import_micro_arm DEPENDS "${MICRO_LUCI_IMPORT_ARM_BINARY}")

nnas_find_package(MbedOSSource EXACT 6.15 QUIET)
nnas_find_package(FlatBuffersSource EXACT 1.10 QUIET)

if (NOT MbedOSSource_FOUND)
    message(STATUS "Skipping luci-micro: MbedOSSource not found")
    return()
endif ()

if (NOT FlatBuffersSource_FOUND)
    message(STATUS "Skipping luci-micro: FlatBuffersSource not found")
    return()
endif ()

set(MBED_OS_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/mbed-os")
file(MAKE_DIRECTORY "${MBED_OS_BUILD_DIR}")

set(MBED_OS_BUILD_DEPENDENCY "${MBED_OS_BUILD_DIR}/CMakeCache.txt")

add_custom_command(
        OUTPUT "${MBED_OS_BUILD_DEPENDENCY}"
        COMMAND "${CMAKE_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/mbed-os"
        -DMICRO_ARM_BUILD_DIR=${MICRO_ARM_BUILD_DIR}
        -DMbedOSSource_DIR=${MbedOSSource_DIR}
        -DLUCI_INTERPRETER_BINARY=${MICRO_ARM_BINARY}
        -DFlatBuffersSource_DIR=${FlatBuffersSource_DIR}
        -DONE_COMPILER_SRC_DIR=${ONE_COMPILER_SRC_DIR}
        WORKING_DIRECTORY "${MBED_OS_BUILD_DIR}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mbed-os/CMakeLists.txt"
        VERBATIM
)

add_custom_target(mbed_os_cmake DEPENDS "${MBED_OS_BUILD_DEPENDENCY}")

set(MBED_OS_BINARY "${MBED_OS_BUILD_DIR}/libmbed_os.a")

add_custom_command(
        OUTPUT "${MBED_OS_BINARY}"
        COMMAND "${CMAKE_MAKE_PROGRAM}" mbed_os -j ${CPU_COUNT}
        WORKING_DIRECTORY "${MBED_OS_BUILD_DIR}"
        DEPENDS mbed_os_cmake
        VERBATIM
)

add_custom_target(mbed_os_arm DEPENDS "${MBED_OS_BINARY}")

set(BENCHMARK_BINARY "${MBED_OS_BUILD_DIR}/benchmark.bin")

add_custom_command(
        OUTPUT "${BENCHMARK_BINARY}"
        COMMAND "${CMAKE_MAKE_PROGRAM}" benchmark -j ${CPU_COUNT}
        WORKING_DIRECTORY "${MBED_OS_BUILD_DIR}"
        DEPENDS mbed_os_arm luci_interpreter_micro_arm luci_import_micro_arm "${CMAKE_CURRENT_SOURCE_DIR}/mbed-os/main.cpp" ${MICRO_ARM_BINARY}
        VERBATIM
)
add_custom_target(luci_micro_benchmark_arm DEPENDS "${BENCHMARK_BINARY}")
