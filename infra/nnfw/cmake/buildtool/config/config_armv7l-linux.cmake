#
# armv7l linux compile options
#

message(STATUS "Building for ARMv7l Linux")

# include linux common
include("cmake/buildtool/config/config_linux.cmake")

# addition for arm-linux
set(FLAGS_COMMON ${FLAGS_COMMON}
    "-mcpu=cortex-a7"
    "-mfloat-abi=hard"
    "-mfpu=neon-vfpv4"
    "-funsafe-math-optimizations"
    "-ftree-vectorize"
    "-mfp16-format=ieee"
    )
