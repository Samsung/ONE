set(ARM_C_COMPILER "arm-none-eabi-gcc")
set(ARM_ASM_COMPILER "arm-none-eabi-gcc")
set(ARM_CXX_COMPILER "arm-none-eabi-g++")
set(ARM_OBJCOPY "arm-none-eabi-objcopy")

find_program(ARM_C_COMPILER_PATH ${ARM_C_COMPILER})

if(NOT ARM_C_COMPILER_PATH)
  message(WARNING "ARM compiler is NOT FOUND, skipping luci-micro build")
  return()
endif()

set(CMAKE_ARM_OPTIONS
  -DLUCI_INTERPRETER_STATIC=ON
  -DLUCI_STATIC=ON
  "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/standalone/Toolchain.cmake"
  "-DLUCI_INTERPRETER_PAL_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../luci-interpreter/pal/mcu"
  "-DNNAS_PROJECT_SOURCE_DIR=${NNAS_PROJECT_SOURCE_DIR}"
  "-DNNAS_EXTERNALS_DIR=${NNAS_EXTERNALS_DIR}"
  -DCPU_ARCH=arm
  -DC_COMPILER=${ARM_C_COMPILER}
  -DCXX_COMPILER=${ARM_CXX_COMPILER}
  -DASM_COMPILER=${ARM_ASM_COMPILER}
  -DOBJCOPY=${ARM_OBJCOPY}
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DENABLE_TEST=OFF
  -DBUILD_GTEST=OFF
  "-DNNAS_ROOT=${NNAS_PROJECT_SOURCE_DIR}"
  -DENABLE_STRICT_BUILD=OFF
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
