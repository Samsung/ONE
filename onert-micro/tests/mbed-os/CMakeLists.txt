CMAKE_MINIMUM_REQUIRED(VERSION 3.9)
SET(CMAKE_SYSTEM_NAME Generic)
SET(CMAKE_CROSSCOMPILING TRUE)

# force compiler settings
SET(CMAKE_C_COMPILER_WORKS TRUE)
SET(CMAKE_CXX_COMPILER_WORKS TRUE)

# force cmake compilers
SET(CMAKE_ASM_COMPILER "arm-none-eabi-gcc")
SET(CMAKE_C_COMPILER "arm-none-eabi-gcc")
SET(CMAKE_CXX_COMPILER "arm-none-eabi-g++")
SET(ELF2BIN "arm-none-eabi-objcopy")


# if the environment does not specify build type, set to Debug
IF (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug"
            CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
            FORCE)
ENDIF ()

# here starts the project
PROJECT(mbed-os-example-blinky C CXX ASM)

# uncomment below to have a verbose build process
#SET(CMAKE_VERBOSE_MAKEFILE ON)

SET(LD_SYS_LIBS "-Wl,--start-group -lstdc++ -lsupc++ -lm -lc -lgcc -lnosys  -Wl,--end-group")


SET(CMAKE_C_FLAGS "-g3 -std=gnu11 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -fmessage-length=0 -fexceptions -ffunction-sections -fdata-sections -funsigned-char -MMD -fomit-frame-pointer -Og -DMBED_DEBUG -DMBED_TRAP_ERRORS_ENABLED=1 -DMBED_MINIMAL_PRINTF -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=softfp -DMBED_ROM_START=0x8000000 -DMBED_ROM_SIZE=0x200000 -DMBED_RAM_START=0x20000400 -DMBED_RAM_SIZE=0x1FC00 -DMBED_RAM1_START=0x24000000 -DMBED_RAM1_SIZE=0x80000 -include ${CMAKE_CURRENT_SOURCE_DIR}/mbed_config.h")
SET(CMAKE_CXX_FLAGS "-g3 -std=gnu++14 -frtti -Wvla -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -fmessage-length=0 -fexceptions -ffunction-sections -fdata-sections -funsigned-char -MMD -fomit-frame-pointer -Og -DMBED_DEBUG -DMBED_TRAP_ERRORS_ENABLED=1 -DMBED_MINIMAL_PRINTF -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=softfp -DMBED_ROM_START=0x8000000 -DMBED_ROM_SIZE=0x200000 -DMBED_RAM_START=0x20000400 -DMBED_RAM_SIZE=0x1FC00 -DMBED_RAM1_START=0x24000000 -DMBED_RAM1_SIZE=0x80000  -include ${CMAKE_CURRENT_SOURCE_DIR}/mbed_config.h")
SET(CMAKE_ASM_FLAGS "-g3 -x assembler-with-cpp -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -fmessage-length=0 -fexceptions -ffunction-sections -fdata-sections -funsigned-char -MMD -fomit-frame-pointer -Og -DMBED_DEBUG -DMBED_TRAP_ERRORS_ENABLED=1 -DMBED_MINIMAL_PRINTF -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=softfp  -include ${CMAKE_CURRENT_SOURCE_DIR}/mbed_config.h")
SET(CMAKE_CXX_LINK_FLAGS "-Wl,--gc-sections -Wl,--wrap,main -Wl,--wrap,__malloc_r -Wl,--wrap,__free_r -Wl,--wrap,__realloc_r -Wl,--wrap,__memalign_r -Wl,--wrap,__calloc_r -Wl,--wrap,exit -Wl,--wrap,atexit -Wl,-n -Wl,--wrap,printf -Wl,--wrap,sprintf -Wl,--wrap,snprintf -Wl,--wrap,vprintf -Wl,--wrap,vsprintf -Wl,--wrap,vsnprintf -Wl,--wrap,fprintf -Wl,--wrap,vfprintf -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=softfp -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -fmessage-length=0 -fexceptions -ffunction-sections -fdata-sections -funsigned-char -MMD -fomit-frame-pointer -Og -DMBED_DEBUG -DMBED_TRAP_ERRORS_ENABLED=1 -DMBED_MINIMAL_PRINTF -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=softfp -DMBED_ROM_START=0x8000000 -DMBED_ROM_SIZE=0x200000 -DMBED_RAM_START=0x20000400 -DMBED_RAM_SIZE=0x1FC00 -DMBED_RAM1_START=0x24000000 -DMBED_RAM1_SIZE=0x80000 -DMBED_BOOT_STACK_SIZE=1024 -DXIP_ENABLE=0 ")
SET(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${LD_SYS_LIBS} -T ${CMAKE_BINARY_DIR}/build_test_pp.link_script.ld")

ADD_DEFINITIONS(
        -DARM_MATH_CM7
        -DCOMPONENT_FLASHIAP=1
        -DDEVICE_ANALOGIN=1
        -DDEVICE_ANALOGOUT=1
        -DDEVICE_CAN=1
        -DDEVICE_CRC=1
        -DDEVICE_EMAC=1
        -DDEVICE_FLASH=1
        -DDEVICE_I2C=1
        -DDEVICE_I2CSLAVE=1
        -DDEVICE_I2C_ASYNCH=1
        -DDEVICE_INTERRUPTIN=1
        -DDEVICE_LPTICKER=1
        -DDEVICE_MPU=1
        -DDEVICE_PORTIN=1
        -DDEVICE_PORTINOUT=1
        -DDEVICE_PORTOUT=1
        -DDEVICE_PWMOUT=1
        -DDEVICE_RESET_REASON=1
        -DDEVICE_RTC=1
        -DDEVICE_SERIAL=1
        -DDEVICE_SERIAL_FC=1
        -DDEVICE_SLEEP=1
        -DDEVICE_SPI=1
        -DDEVICE_SPISLAVE=1
        -DDEVICE_SPI_ASYNCH=1
        -DDEVICE_STDIO_MESSAGES=1
        -DDEVICE_TRNG=1
        -DDEVICE_USBDEVICE=1
        -DDEVICE_USTICKER=1
        -DDEVICE_WATCHDOG=1
        -DEXTRA_IDLE_STACK_REQUIRED
        -DMBED_BUILD_TIMESTAMP=1640167847.81
        -DMBED_TICKLESS
        -DSTM32H743xx
        -DTARGET_CORTEX
        -DTARGET_CORTEX_M
        -DTARGET_FF_ARDUINO_UNO
        -DTARGET_LIKE_CORTEX_M7
        -DTARGET_LIKE_MBED
        -DTARGET_M7
        -DTARGET_MCU_STM32
        -DTARGET_MCU_STM32H7
        -DTARGET_MCU_STM32H743xI
        -DTARGET_NAME=NUCLEO_H743ZI2
        -DTARGET_NUCLEO_H743ZI2
        -DTARGET_RELEASE
        -DTARGET_RTOS_M4_M7
        -DTARGET_STM
        -DTARGET_STM32H7
        -DTARGET_STM32H743xI
        -DTOOLCHAIN_GCC
        -DTOOLCHAIN_GCC_ARM
        -DTRANSACTION_QUEUE_SIZE_SPI=2
        -DUSE_FULL_LL_DRIVER
        -DUSE_HAL_DRIVER
        -D__CMSIS_RTOS
        -D__CORTEX_M7
        -D__FPU_PRESENT=1
        -D__MBED_CMSIS_RTOS_CM
        -D__MBED__=1
)

include(mbed-sources.cmake)

set_sources_mbed(${MbedOSSource_DIR})

add_library(mbed_os STATIC ${SOURCES})

target_include_directories_mbed(mbed_os ${MbedOSSource_DIR})


SET_TARGET_PROPERTIES(mbed_os PROPERTIES ENABLE_EXPORTS 1)
# add syslibs dependencies to create the correct linker order
TARGET_LINK_LIBRARIES(mbed_os -lstdc++ -lsupc++ -lm -lc -lgcc -lnosys)

add_executable(build_test main.cpp)

target_link_libraries(build_test mbed_os)
target_include_directories_mbed(build_test ${MbedOSSource_DIR})

target_link_libraries(mbed_os "${MICRO_ARM_BUILD_DIR}/luci-interpreter/src/core/reader/libluci_micro_circle_reader.a")
target_link_libraries(mbed_os "${MICRO_ARM_BUILD_DIR}/luci-interpreter/src/core/libluci_interpreter_core_micro.a")
target_link_libraries(mbed_os "${MICRO_ARM_BUILD_DIR}/luci-interpreter/src/kernels/libluci_interpreter_kernels_micro.a")
target_link_libraries(mbed_os "${MICRO_ARM_BUILD_DIR}/luci-interpreter/src/kernels/libluci_interpreter_mcu_pal.a")
target_link_libraries(mbed_os "${MICRO_ARM_BUILD_DIR}/luci-interpreter/src/loader/libluci_interpreter_loader_micro.a")
target_link_libraries(mbed_os "${MICRO_ARM_BUILD_DIR}/luci-interpreter/src/libluci_interpreter_micro.a")
target_link_libraries(mbed_os "${MICRO_ARM_BUILD_DIR}/luci-interpreter/src/memory_managers/libluci_interpreter_micro_memory_manager.a")

target_link_libraries(build_test "${MICRO_ARM_BUILD_DIR}/luci-interpreter/src/core/reader/libluci_micro_circle_reader.a")
target_link_libraries(build_test "${MICRO_ARM_BUILD_DIR}/luci-interpreter/src/core/libluci_interpreter_core_micro.a")
target_link_libraries(build_test "${MICRO_ARM_BUILD_DIR}/luci-interpreter/src/kernels/libluci_interpreter_kernels_micro.a")
target_link_libraries(build_test "${MICRO_ARM_BUILD_DIR}/luci-interpreter/src/kernels/libluci_interpreter_mcu_pal.a")
target_link_libraries(build_test "${MICRO_ARM_BUILD_DIR}/luci-interpreter/src/loader/libluci_interpreter_loader_micro.a")
target_link_libraries(build_test "${MICRO_ARM_BUILD_DIR}/luci-interpreter/src/libluci_interpreter_micro.a")
target_link_libraries(build_test "${MICRO_ARM_BUILD_DIR}/luci-interpreter/src/memory_managers/libluci_interpreter_micro_memory_manager.a")

target_include_directories(build_test PUBLIC
        ${ONERTMICRO_SRC_DIR}/luci-interpreter/include
        ${ONERTMICRO_SRC_DIR}/luci-interpreter/src
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${FlatBuffersSource_DIR}/include
        )

add_custom_command(TARGET build_test PRE_LINK
        COMMAND "arm-none-eabi-cpp" -E -P -Wl,--gc-sections -Wl,--wrap,main -Wl,--wrap,_malloc_r -Wl,--wrap,_free_r -Wl,--wrap,_realloc_r -Wl,--wrap,_memalign_r -Wl,--wrap,_calloc_r -Wl,--wrap,exit -Wl,--wrap,atexit -Wl,-n -Wl,--wrap,printf -Wl,--wrap,sprintf -Wl,--wrap,snprintf -Wl,--wrap,vprintf -Wl,--wrap,vsprintf -Wl,--wrap,vsnprintf -Wl,--wrap,fprintf -Wl,--wrap,vfprintf -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=softfp -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -fmessage-length=0 -fexceptions -ffunction-sections -fdata-sections -funsigned-char -MMD -fomit-frame-pointer -Og -DMBED_DEBUG -DMBED_TRAP_ERRORS_ENABLED=1 -DMBED_MINIMAL_PRINTF -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=softfp -DMBED_ROM_START=0x8000000 -DMBED_ROM_SIZE=0x200000 -DMBED_RAM_START=0x20000400 -DMBED_RAM_SIZE=0x1FC00 -DMBED_RAM1_START=0x24000000 -DMBED_RAM1_SIZE=0x80000 -DMBED_BOOT_STACK_SIZE=1024 -DXIP_ENABLE=0 ${MbedOSSource_DIR}/targets/TARGET_STM/TARGET_STM32H7/TARGET_STM32H743xI/TOOLCHAIN_GCC_ARM/STM32H743xI.ld -o ${CMAKE_CURRENT_BINARY_DIR}/build_test_pp.link_script.ld
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/build_test_pp.link_script.ld"
        )

add_custom_command(TARGET build_test POST_BUILD
        COMMAND ${ELF2BIN} -O binary $<TARGET_FILE:build_test> $<TARGET_FILE:build_test>.bin
        COMMAND ${CMAKE_COMMAND} -E echo "-- built: $<TARGET_FILE:build_test>.bin"
        )
