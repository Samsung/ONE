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
    )

# "fp16-format=ieee" is default and not supported flag on Clang
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-mfp16-format=ieee COMPILER_SUPPORTS_FP16_FORMAT_I3E)
if(COMPILER_SUPPORTS_FP16_FORMAT_I3E)
    set(FLAGS_COMMON ${FLAGS_COMMON} "-mfp16-format=ieee")
endif()
