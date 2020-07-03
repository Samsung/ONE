#
# x86_64 darwin(macOS) compile options
#
message(STATUS "Building for x86-64 Darwin")

# SIMD for x86
set(FLAGS_COMMON ${FLAGS_COMMON}
    "-msse4"
    )

# lib pthread as a variable (pthread must be disabled on android)
set(LIB_PTHREAD pthread)
