include("cmake/buildtool/config/config_linux.cmake")

# On Android, pthread is contained in bionic(libc)
set(LIB_PTHREAD "")

# SIMD for aarch64
set(FLAGS_COMMON ${FLAGS_COMMON}
    "-ftree-vectorize"
    "-DUSE_RUY_GEMV"
    )
