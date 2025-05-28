#
# armv7l tizen compile options
#

message(STATUS "Building for ARMv7hl(hardfp) Tizen")

# Build flag for tizen
set(CMAKE_C_FLAGS_DEBUG     "-O -g -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O -g -DDEBUG")

# TODO : add and use option_tizen if something uncommon comes up
# include linux common
include("${CMAKE_CURRENT_LIST_DIR}/config_linux.cmake")

# addition for arm-linux
set(FLAGS_COMMON ${FLAGS_COMMON}
    "-mfpu=neon-vfpv4"
    "-funsafe-math-optimizations"
    "-ftree-vectorize"
    "-mfp16-format=ieee"
    )
