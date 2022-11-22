#
# armv7l linux compile options
#

message(STATUS "Building for ARMv7l Linux")

# include linux common
include("${CMAKE_CURRENT_LIST_DIR}/config_linux.cmake")

# addition for arm-linux
set(FLAGS_COMMON ${FLAGS_COMMON}
    "-march=armv7-a"
    "-mtune=cortex-a8"
    "-mfloat-abi=hard"
    "-mfp16-format=ieee"
    )

if(BUILD_ARM32_NEON)
  set(FLAGS_COMMON ${FLAGS_COMMON}
      "-mfpu=vfpv3-d16"
      "-ftree-vectorize"
      )
else(BUILD_ARM32_NEON)
  message(STATUS "ARMv7l: NEON is disabled")
endif(BUILD_ARM32_NEON)
