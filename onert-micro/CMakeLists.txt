#set(ARM_C_COMPILER "arm-none-eabi-gcc")
#set(ARM_ASM_COMPILER "arm-none-eabi-gcc")
#set(ARM_CXX_COMPILER "arm-none-eabi-g++")
#set(ARM_OBJCOPY "arm-none-eabi-objcopy")
#
#find_program(ARM_C_COMPILER_PATH ${ARM_C_COMPILER})
#
#if (NOT ARM_C_COMPILER_PATH)
#    message(STATUS "Build onert-micro failed: ARM compiler is NOT FOUND")
#    return()
#endif ()
#
#set(OM_CIRCLE_SCHEMA onert_micro_circle_schema)
#
#if (NOT_BUILD_EXTERNALS)
#    message(STATUS "USE LOCAL EXTERNAL")
#
#    set(ONERT_MICRO_EXTERNAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals")
#    set(Flatbuffers_DIR "${ONERT_MICRO_EXTERNAL_DIR}")
#    set(FlatBuffersSource_DIR "${Flatbuffers_DIR}")
#    set (EXT_OVERLAY_DIR "${ONERT_MICRO_EXTERNAL_DIR}")
#    set(GENERATED_INCLUDE_DIR "${ONERT_MICRO_EXTERNAL_DIR}/gen")
#
#    add_subdirectory(externals)
#else()
#    message(STATUS "DOWNLOAD AND BUILD EXTERNALS")
#
#    nnas_find_package(FlatBuffers EXACT 2.0 QUIET)
#
#    if (NOT FlatBuffers_FOUND)
#        message(STATUS "Build onert-micro failed: FlatBuffers 2.0 NOT FOUND")
#        return()
#    endif (NOT FlatBuffers_FOUND)
#
#    message(STATUS "FOUND FlatBuffers")
#
#    set(SCHEMA_FILE "${NNAS_PROJECT_SOURCE_DIR}/res/CircleSchema/0.6/circle_schema.fbs")
#
#    # NOTE Copy circle_schema.fbs as schema.fbs to generate "schema_generated.fbs" instead of "circle_schema_generated.fbs"
#    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/schema.fbs"
#            COMMAND ${CMAKE_COMMAND} -E copy "${SCHEMA_FILE}" schema.fbs
#            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
#            DEPENDS "${SCHEMA_FILE}"
#            )
#
#    FlatBuffers_Target(${OM_CIRCLE_SCHEMA}
#            OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen/circle-generated/circle"
#            INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen"
#            SCHEMA_DIR "${CMAKE_CURRENT_BINARY_DIR}"
#            SCHEMA_FILES "schema.fbs"
#            )
#    set(Flatbuffers_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../overlay/lib/cmake/flatbuffers")
#    set(EXT_OVERLAY_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../overlay")
#    set(GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen")
#endif()
#
##set(LUCI_INTERPRETER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/luci-interpreter/include")
##add_subdirectory(luci-interpreter/src/core/reader)
#
## Choosing Kernel: reference mcu, optimized cmsisnn
#if (NOT KERNELS)
#    message(STATUS "KERNEL variable is not defined, default reference mcu kernels will be used")
#    set(OM_PAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onert-micro/include/execute/pal/mcu")
#elseif("${KERNELS}" STREQUAL "mcu")
#    message(STATUS "ONERT_MICRO will use reference mcu kernels")
#    set(OM_PAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onert-micro/include/execute/pal/mcu")
#elseif("${KERNELS}" STREQUAL "cmsisnn")
#    message(STATUS "ONERT_MICRO will use optimized cmsisnn kernels")
#    set(OM_PAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onert-micro/include/execute/pal/cmsisnn")
#else()
#    message(STATUS "Build onert-micro: FAILED (Non-existent kernel variable. Choose one of the following options: mcu, cmsisnn)")
#    return()
#endif()
#
#set(CMAKE_ARM_OPTIONS
#        -DTARGET_CPU=${TARGET_CPU}
#        -DTARGET_ARCH=${TARGET_ARCH}
#        "-DEXT_OVERLAY_DIR=${EXT_OVERLAY_DIR}"
#        "-DFlatBuffersSource_DIR=${FlatBuffersSource_DIR}"
#        "-DCMAKE_TOOLCHAIN_FILE=${NNAS_PROJECT_SOURCE_DIR}/infra/onert-micro/cmake/buildtool/config/arm-none-eabi-gcc.cmake"
#        "-DOM_PAL_DIR=${OM_PAL_DIR}"
#        "-DNNAS_PROJECT_SOURCE_DIR=${NNAS_PROJECT_SOURCE_DIR}"
#        "-DNNAS_EXTERNALS_DIR=${NNAS_EXTERNALS_DIR}"
#        -DC_COMPILER=${ARM_C_COMPILER}
#        -DCXX_COMPILER=${ARM_CXX_COMPILER}
#        -DASM_COMPILER=${ARM_ASM_COMPILER}
#        -DOBJCOPY=${ARM_OBJCOPY}
#        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
#        -DENABLE_TEST=OFF
#        -DBUILD_GTEST=OFF
#        "-DNNAS_ROOT=${NNAS_PROJECT_SOURCE_DIR}"
#        "-DGENERATED_INCLUDE_DIR=${GENERATED_INCLUDE_DIR}"
#        )
#
#if (GENERATE_KERNELS_LIST_FROM)
#    set(GENERATED_KERNELS_LIST_PATH "${LUCI_INTERPRETER_PAL_DIR}/GeneratedKernelsToBuild.lst")
#    list(APPEND CMAKE_ARM_OPTIONS "-DLUCI_INTERPRETER_KERNELS_BUILD_LIST=${GENERATED_KERNELS_LIST_PATH}")
#endif ()
#
#if (NOT_BUILD_EXTERNALS)
#    list(APPEND CMAKE_ARM_OPTIONS "-DNOT_BUILD_EXTERNALS=ON")
#else()
#    list(APPEND CMAKE_ARM_OPTIONS "-DNOT_BUILD_EXTERNALS=OFF")
#endif ()
#
#if (ENABLE_ONERT_MICRO_TEST)
#    message(STATUS "Enable tests")
#    list(APPEND CMAKE_ARM_OPTIONS "-DENABLE_TEST=ON")
#endif ()
#
#if (ENABLE_ONERT_MICRO_TRAINING)
#    message(STATUS "ENABLE TRAINING PART")
#    add_definitions(-DENABLE_TRAINING)
#    list(APPEND CMAKE_ARM_OPTIONS "-DENABLE_TRAINING=ON")
#endif ()
#
#if (DIS_QUANT)
#    message(STATUS "ONERT-MICRO will not use part for QUANTIZED models")
#    add_definitions(-DDIS_QUANT)
#    list(APPEND CMAKE_ARM_OPTIONS "-DDIS_QUANT=ON")
#endif()
#
#if (DIS_FLOAT)
#    message(STATUS "ONERT-MICRO will not use part for FLOAT models")
#    add_definitions(-DDIS_FLOAT)
#    list(APPEND CMAKE_ARM_OPTIONS "-DDIS_FLOAT=ON")
#endif()
#
#if (DIS_DYN_SHAPES)
#    message(STATUS "ONERT-MICRO will not use dynamic shapes")
#    add_definitions(-DDIS_DYN_SHAPES)
#endif()
#
#set(MICRO_ARM_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/standalone_arm")
#file(MAKE_DIRECTORY "${MICRO_ARM_BUILD_DIR}")
#
#set(MICRO_ARM_BUILD_DEPENDENCY "${MICRO_ARM_BUILD_DIR}/CMakeCache.txt")
#
#add_custom_command(
#        OUTPUT "${MICRO_ARM_BUILD_DEPENDENCY}"
#        COMMAND "${CMAKE_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/standalone" ${CMAKE_ARM_OPTIONS}
#        WORKING_DIRECTORY "${MICRO_ARM_BUILD_DIR}"
#        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/standalone/CMakeLists.txt"
#        VERBATIM
#)
#
#add_custom_target(onert_micro_arm_cmake DEPENDS "${MICRO_ARM_BUILD_DEPENDENCY}")
#
### Generate KernelsToBuild list from circle model
##if (GENERATE_KERNELS_LIST_FROM)
##    add_executable(generator_kernels_list_exec helpers/GenerateKernelsListHelper.cpp)
##
##    target_link_libraries(generator_kernels_list_exec ${OM_CIRCLE_SCHEMA})
##    target_link_libraries(generator_kernels_list_exec ${OM_CIRCLE_SCHEMA})
##
##    target_include_directories(generator_kernels_list_exec PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/luci-interpreter/include")
##
##    add_custom_target(
##            generate_kernels_list ALL
##            COMMAND generator_kernels_list_exec ${GENERATE_KERNELS_LIST_FROM} ${GENERATED_KERNELS_LIST_PATH}
##            COMMENT "Generating KernelsToBuild list"
##    )
##    add_dependencies(generate_kernels_list luci_micro_circle_reader)
##    add_dependencies(luci_interpreter_micro_arm_cmake generate_kernels_list)
##
##endif ()
#
## To remove GENERATE_KERNELS_LIST_FROM and KERNELS variable from cmake cache
#unset(GENERATE_KERNELS_LIST_FROM CACHE)
#unset(KERNELS CACHE)
#unset(USE_STATIC_KERNEL CACHE)
#unset(DIS_QUANT CACHE)
#unset(DIS_FLOAT CACHE)
#unset(ENABLE_ONERT_MICRO_TEST CACHE)
#unset(NOT_BUILD_EXTERNALS CACHE)
#
#set(MICRO_ARM_BINARY "${MICRO_ARM_BUILD_DIR}/luci-interpreter/src/libluci_interpreter_micro.a")
#
#add_custom_command(
#        OUTPUT "${MICRO_ARM_BINARY}"
#        COMMAND "${CMAKE_MAKE_PROGRAM}" luci_interpreter_micro -j ${CPU_COUNT}
#        WORKING_DIRECTORY "${MICRO_ARM_BUILD_DIR}"
#        DEPENDS onert_micro_arm_cmake
#        VERBATIM
#)
#
#add_custom_target(luci_interpreter_micro_arm DEPENDS "${MICRO_ARM_BINARY}")
#
##add_subdirectory(eval-driver)
#
## Should be after add_subdirectory
#unset(ENABLE_ONERT_MICRO_TRAINING CACHE)
#
#if (NOT DEFINED BUILD_TEST)
#    return()
#endif ()
#
##MBED OS QEMU build
#nnas_find_package(MbedOSSource EXACT 6.15 QUIET)
#
#if (NOT MbedOSSource_FOUND)
#    message(STATUS "Skipping luci-micro: MbedOSSource not found")
#    return()
#endif ()
#
#set(MBED_OS_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/tests/mbed-os")
#file(MAKE_DIRECTORY "${MBED_OS_BUILD_DIR}")
#
#set(MBED_OS_BUILD_DEPENDENCY "${MBED_OS_BUILD_DIR}/CMakeCache.txt")
#
#set(ONERTMICRO_SRC_DIR "${NNAS_PROJECT_SOURCE_DIR}/onert-micro")
#
#add_custom_command(
#        OUTPUT "${MBED_OS_BUILD_DEPENDENCY}"
#        COMMAND "${CMAKE_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/mbed-os"
#        -DMICRO_ARM_BUILD_DIR=${MICRO_ARM_BUILD_DIR}
#        -DMbedOSSource_DIR=${MbedOSSource_DIR}
#        -DFlatBuffersSource_DIR=${FlatBuffersSource_DIR}
#        -DONERTMICRO_SRC_DIR=${ONERTMICRO_SRC_DIR}
#        WORKING_DIRECTORY "${MBED_OS_BUILD_DIR}"
#        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/mbed-os/CMakeLists.txt"
#        VERBATIM
#)
#
#add_custom_target(mbed_os_cmake DEPENDS "${MBED_OS_BUILD_DEPENDENCY}")
#
#set(MBED_OS_BINARY "${MBED_OS_BUILD_DIR}/libmbed_os.a")
#
#add_custom_command(
#        OUTPUT "${MBED_OS_BINARY}"
#        COMMAND "${CMAKE_MAKE_PROGRAM}" mbed_os -j ${CPU_COUNT}
#        WORKING_DIRECTORY "${MBED_OS_BUILD_DIR}"
#        DEPENDS mbed_os_cmake
#        VERBATIM
#)
#
#add_custom_target(mbed_os_arm DEPENDS "${MBED_OS_BINARY}")
#
#set(BUILD_TEST_BINARY "${MBED_OS_BUILD_DIR}/build_test.bin")
#
#add_custom_command(
#        OUTPUT "${BUILD_TEST_BINARY}"
#        COMMAND "${CMAKE_MAKE_PROGRAM}" build_test -j ${CPU_COUNT}
#        WORKING_DIRECTORY "${MBED_OS_BUILD_DIR}"
#        DEPENDS mbed_os_arm "${CMAKE_CURRENT_SOURCE_DIR}/tests/mbed-os/main.cpp" ${MICRO_ARM_BINARY}
#        VERBATIM
#)
#add_custom_target(onert_micro_build_test_arm DEPENDS "${BUILD_TEST_BINARY}")
















#
set(ARM_C_COMPILER "arm-none-eabi-gcc")
set(ARM_ASM_COMPILER "arm-none-eabi-gcc")
set(ARM_CXX_COMPILER "arm-none-eabi-g++")
set(ARM_OBJCOPY "arm-none-eabi-objcopy")

find_program(ARM_C_COMPILER_PATH ${ARM_C_COMPILER})

if (NOT ARM_C_COMPILER_PATH)
    message(STATUS "Build onert-micro failed: ARM compiler is NOT FOUND")
    return()
endif ()

set(OM_CIRCLE_SCHEMA onert_micro_circle_schema)

if (NOT_BUILD_EXTERNALS)
    message(STATUS "USE LOCAL EXTERNAL")

    set(ONERT_MICRO_EXTERNAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals")
    set(Flatbuffers_DIR "${ONERT_MICRO_EXTERNAL_DIR}")
    set(FlatBuffersSource_DIR "${Flatbuffers_DIR}")
    set (EXT_OVERLAY_DIR "${ONERT_MICRO_EXTERNAL_DIR}")
    set(GENERATED_INCLUDE_DIR "${ONERT_MICRO_EXTERNAL_DIR}/gen")

    add_subdirectory(externals)
else()
    message(STATUS "DOWNLOAD AND BUILD EXTERNALS")

    nnas_find_package(FlatBuffers EXACT 2.0 QUIET)

    if (NOT FlatBuffers_FOUND)
        message(STATUS "Build onert-micro failed: FlatBuffers 2.0 NOT FOUND")
        return()
    endif (NOT FlatBuffers_FOUND)

    message(STATUS "FOUND FlatBuffers")

    set(SCHEMA_FILE "${NNAS_PROJECT_SOURCE_DIR}/res/CircleSchema/0.6/circle_schema.fbs")

    # NOTE Copy circle_schema.fbs as schema.fbs to generate "schema_generated.fbs" instead of "circle_schema_generated.fbs"
    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/schema.fbs"
            COMMAND ${CMAKE_COMMAND} -E copy "${SCHEMA_FILE}" schema.fbs
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            DEPENDS "${SCHEMA_FILE}"
            )

    FlatBuffers_Target(${OM_CIRCLE_SCHEMA}
            OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen/circle-generated/circle"
            INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen"
            SCHEMA_DIR "${CMAKE_CURRENT_BINARY_DIR}"
            SCHEMA_FILES "schema.fbs"
            )
    set(Flatbuffers_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../overlay/lib/cmake/flatbuffers")
    set(EXT_OVERLAY_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../overlay")
    set(GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen")
endif()

#set(LUCI_INTERPRETER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/luci-interpreter/include")
#add_subdirectory(luci-interpreter/src/core/reader)

# Choosing Kernel: reference mcu, optimized cmsisnn
if (NOT KERNELS)
    message(STATUS "KERNEL variable is not defined, default reference mcu kernels will be used")
    set(OM_PAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onert-micro/include/pal/mcu")
elseif("${KERNELS}" STREQUAL "mcu")
    message(STATUS "ONERT_MICRO will use reference mcu kernels")
    set(OM_PAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onert-micro/include/pal/mcu")
elseif("${KERNELS}" STREQUAL "cmsisnn")
    message(STATUS "ONERT_MICRO will use optimized cmsisnn kernels")
    set(OM_PAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onert-micro/include/pal/cmsisnn")
else()
    message(STATUS "Build onert-micro: FAILED (Non-existent kernel variable. Choose one of the following options: mcu, cmsisnn)")
    return()
endif()

set(CMAKE_ARM_OPTIONS
        -DTARGET_CPU=${TARGET_CPU}
        -DTARGET_ARCH=${TARGET_ARCH}
        "-DEXT_OVERLAY_DIR=${EXT_OVERLAY_DIR}"
        "-DFlatBuffersSource_DIR=${FlatBuffersSource_DIR}"
        "-DCMAKE_TOOLCHAIN_FILE=${NNAS_PROJECT_SOURCE_DIR}/infra/onert-micro/cmake/buildtool/config/arm-none-eabi-gcc.cmake"
        "-DOM_PAL_DIR=${OM_PAL_DIR}"
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
        "-DGENERATED_INCLUDE_DIR=${GENERATED_INCLUDE_DIR}"
        )
#
#if (GENERATE_KERNELS_LIST_FROM)
#    set(GENERATED_KERNELS_LIST_PATH "${LUCI_INTERPRETER_PAL_DIR}/GeneratedKernelsToBuild.lst")
#    list(APPEND CMAKE_ARM_OPTIONS "-DLUCI_INTERPRETER_KERNELS_BUILD_LIST=${GENERATED_KERNELS_LIST_PATH}")
#endif ()

if (NOT_BUILD_EXTERNALS)
    list(APPEND CMAKE_ARM_OPTIONS "-DNOT_BUILD_EXTERNALS=ON")
else()
    list(APPEND CMAKE_ARM_OPTIONS "-DNOT_BUILD_EXTERNALS=OFF")
endif ()

if (ENABLE_ONERT_MICRO_TEST)
    message(STATUS "Enable tests")
    list(APPEND CMAKE_ARM_OPTIONS "-DENABLE_TEST=ON")
endif ()
set(ENABLE_ONERT_MICRO_TRAINING 1)
if (ENABLE_ONERT_MICRO_TRAINING)
    message(STATUS "ENABLE TRAINING PART")
    add_definitions(-DENABLE_TRAINING)
    list(APPEND CMAKE_ARM_OPTIONS "-DENABLE_TRAINING=ON")
endif ()

if (DIS_QUANT)
    message(STATUS "ONERT-MICRO will not use part for QUANTIZED models")
    add_definitions(-DDIS_QUANT)
    list(APPEND CMAKE_ARM_OPTIONS "-DDIS_QUANT=ON")
endif()

if (DIS_FLOAT)
    message(STATUS "ONERT-MICRO will not use part for FLOAT models")
    add_definitions(-DDIS_FLOAT)
    list(APPEND CMAKE_ARM_OPTIONS "-DDIS_FLOAT=ON")
endif()

if (DIS_DYN_SHAPES)
    message(STATUS "ONERT-MICRO will not use dynamic shapes")
    add_definitions(-DDIS_DYN_SHAPES)
endif()

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

add_custom_target(onert_micro_arm_cmake DEPENDS "${MICRO_ARM_BUILD_DEPENDENCY}")

## Generate KernelsToBuild list from circle model
#if (GENERATE_KERNELS_LIST_FROM)
#    add_executable(generator_kernels_list_exec helpers/GenerateKernelsListHelper.cpp)
#
#    target_link_libraries(generator_kernels_list_exec ${OM_CIRCLE_SCHEMA})
#    target_link_libraries(generator_kernels_list_exec ${OM_CIRCLE_SCHEMA})
#
#    target_include_directories(generator_kernels_list_exec PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/luci-interpreter/include")
#
#    add_custom_target(
#            generate_kernels_list ALL
#            COMMAND generator_kernels_list_exec ${GENERATE_KERNELS_LIST_FROM} ${GENERATED_KERNELS_LIST_PATH}
#            COMMENT "Generating KernelsToBuild list"
#    )
#    add_dependencies(generate_kernels_list luci_micro_circle_reader)
#    add_dependencies(luci_interpreter_micro_arm_cmake generate_kernels_list)
#
#endif ()

# To remove GENERATE_KERNELS_LIST_FROM and KERNELS variable from cmake cache
unset(GENERATE_KERNELS_LIST_FROM CACHE)
unset(KERNELS CACHE)
unset(USE_STATIC_KERNEL CACHE)
unset(DIS_QUANT CACHE)
unset(DIS_FLOAT CACHE)
unset(ENABLE_ONERT_MICRO_TEST CACHE)
unset(NOT_BUILD_EXTERNALS CACHE)

set(MICRO_ARM_BINARY "${MICRO_ARM_BUILD_DIR}/onert_micro/src/libonert_micro_interpreter.a")

add_custom_command(
        OUTPUT "${MICRO_ARM_BINARY}"
        COMMAND "${CMAKE_MAKE_PROGRAM}" onert_micro_interpreter -j ${CPU_COUNT}
        WORKING_DIRECTORY "${MICRO_ARM_BUILD_DIR}"
        DEPENDS onert_micro_arm_cmake ${OM_CIRCLE_SCHEMA}
        VERBATIM
)

add_custom_target(onert_micro_arm DEPENDS "${MICRO_ARM_BINARY}")

add_subdirectory(eval-driver)

# Should be after add_subdirectory
unset(ENABLE_ONERT_MICRO_TRAINING CACHE)

# Example build
if (NOT DEFINED BUILD_EXAMPLE)
    return()
endif ()

#MBED OS QEMU build
nnas_find_package(MbedOSSource EXACT 6.15 QUIET)

if (NOT MbedOSSource_FOUND)
    message(STATUS "Skipping luci-micro: MbedOSSource not found")
    return()
endif ()

set(MBED_OS_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/examples/speech_recognition/DISCO_F746/mbed-os")
file(MAKE_DIRECTORY "${MBED_OS_BUILD_DIR}")

set(MBED_OS_BUILD_DEPENDENCY "${MBED_OS_BUILD_DIR}/CMakeCache.txt")

set(ONERTMICRO_SRC_DIR "${NNAS_PROJECT_SOURCE_DIR}/onert-micro")

add_custom_command(
        OUTPUT "${MBED_OS_BUILD_DEPENDENCY}"
        COMMAND "${CMAKE_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/examples/speech_recognition/DISCO_F746/mbed-os"
        -DMICRO_ARM_BUILD_DIR=${MICRO_ARM_BUILD_DIR}
        -DMbedOSSource_DIR=${MbedOSSource_DIR}
        -DFlatBuffersSource_DIR=${FlatBuffersSource_DIR}
        -DONERTMICRO_SRC_DIR=${ONERTMICRO_SRC_DIR}
        WORKING_DIRECTORY "${MBED_OS_BUILD_DIR}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/examples/speech_recognition/DISCO_F746/mbed-os/CMakeLists.txt"
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

set(BUILD_EXAMPLE_BINARY "${MBED_OS_BUILD_DIR}/example_speech_recognition.bin")

add_custom_command(
        OUTPUT "${BUILD_EXAMPLE_BINARY}"
        COMMAND "${CMAKE_MAKE_PROGRAM}" example_speech_recognition -j ${CPU_COUNT}
        WORKING_DIRECTORY "${MBED_OS_BUILD_DIR}"
        DEPENDS mbed_os_arm "${CMAKE_CURRENT_SOURCE_DIR}/examples/speech_recognition/DISCO_F746/mbed-os/main.cpp" ${MICRO_ARM_BINARY}
        VERBATIM
)
add_custom_target(onert_micro_build_example_arm DEPENDS "${BUILD_EXAMPLE_BINARY}")
