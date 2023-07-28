#
# armv7l tizen compile options
#

message(STATUS "Building for ARMv7hl(hardfp) Tizen")

# Build flag for tizen
set(CMAKE_C_FLAGS_DEBUG     "-O -g -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O -g -DDEBUG")

# TODO : add and use option_tizen if something uncommon comes up
# include linux common
include("cmake/buildtool/config/config_linux.cmake")

# addition for arm-linux
set(FLAGS_COMMON ${FLAGS_COMMON}
    "-mtune=cortex-a8"
    "-mfloat-abi=hard"
    "-funsafe-math-optimizations"
    )

if(BUILD_ARM32_NEON)
    set(FLAGS_COMMON ${FLAGS_COMMON}
        "-mfpu=neon-vfpv4"
        "-ftree-vectorize"
    )
else(BUILD_ARM32_NEON)
    message(STATUS "ARMv7l: NEON is disabled")
endif(BUILD_ARM32_NEON)
