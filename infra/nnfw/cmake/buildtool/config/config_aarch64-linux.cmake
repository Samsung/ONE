#
# aarch64 linux compile options
#

message(STATUS "Building for AARCH64 Linux")

# include linux common
include("cmake/buildtool/config/config_linux.cmake")

# addition for aarch64-linux
set(FLAGS_COMMON ${FLAGS_COMMON})
