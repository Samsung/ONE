#
# armv7l linux compile options
#

message(STATUS "Building for ARMv7l Linux")

# include linux common
include("${CMAKE_CURRENT_LIST_DIR}/config_linux.cmake")

# addition for arm-linux
set(FLAGS_COMMON ${FLAGS_COMMON}
    "-march=armv7-a"
    "-mtune=cortex-a15.cortex-a7"
    "-mfloat-abi=hard"
    "-mfpu=neon-vfpv4"
    "-ftree-vectorize"
    "-mfp16-format=ieee"
    )
