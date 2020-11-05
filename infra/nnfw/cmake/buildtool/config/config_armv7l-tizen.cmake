#
# armv7l tizen compile options
#

message(STATUS "Building for ARMv7l(softfp) Tizen")

# Build flag for tizen
set(CMAKE_C_FLAGS_DEBUG     "-O -g -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O -g -DDEBUG")

# TODO : add and use option_tizen if something uncommon comes up
# include linux common
include("cmake/buildtool/config/config_linux.cmake")

# addition for arm-linux
set(FLAGS_COMMON ${FLAGS_COMMON}
    "-mtune=cortex-a8"
    "-mfloat-abi=softfp"
    "-funsafe-math-optimizations"
    "-ftree-vectorize"
    )

if(VFPV3_BUILD)
    set(FLAGS_COMMON ${FLAGS_COMMON} "-mfpu=neon-vfpv3")
else(VFPV3_BUILD)
    set(FLAGS_COMMON ${FLAGS_COMMON} "-mfpu=neon-vfpv4")
endif(VFPV3_BUILD)
