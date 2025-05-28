#
# armv7l tizen compile options
#

message(STATUS "Building for ARMv7l(softfp) Tizen")

# Build flag for tizen
set(CMAKE_C_FLAGS_DEBUG     "-O -g -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O -g -DDEBUG")

# TODO : add and use option_tizen if something uncommon comes up
# include linux common
include("${CMAKE_CURRENT_LIST_DIR}/config_linux.cmake")

# addition for arm-linux
set(MFPU_OPTION "-mfpu=neon-vfpv4")
string(REGEX MATCH "-march=([a-zA-Z0-9]+)" MARCH_MATCH "${CMAKE_C_FLAGS}")
if(MARCH_MATCH)
    string(REGEX REPLACE "-march=" "" MARCH_VALUE "${MARCH_MATCH}")
    # Check if the target arch is armv8
    if(MARCH_VALUE MATCHES "armv8")
        set(MFPU_OPTION "-mfpu=neon-fp-armv8")
    endif()
endif(MARCH_MATCH)
set(FLAGS_COMMON ${FLAGS_COMMON}
    ${MFPU_OPTION}
    "-funsafe-math-optimizations"
    "-ftree-vectorize"
    "-mfp16-format=ieee"
    )
