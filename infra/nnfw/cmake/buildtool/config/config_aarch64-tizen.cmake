#
# aarch64 tizen compile options
#

message(STATUS "Building for AARCH64 Tizen")

# Build flag for tizen
set(CMAKE_C_FLAGS_DEBUG     "-O -g -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O -g -DDEBUG")

# TODO : add and use option_tizen if something uncommon comes up
# include linux common
include("cmake/buildtool/config/config_linux.cmake")

# addition for aarch64-tizen
set(FLAGS_COMMON ${FLAGS_COMMON}
    )
