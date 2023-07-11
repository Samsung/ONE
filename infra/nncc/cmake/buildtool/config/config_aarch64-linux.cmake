#
# aarch64 linux compile options
#

message(STATUS "Building for aarch64 Linux")

# include linux common
include("${CMAKE_CURRENT_LIST_DIR}/config_linux.cmake")

# addition for arm-linux
set(FLAGS_COMMON ${FLAGS_COMMON}
    "-march=armv8-a"
    )
