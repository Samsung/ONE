include("${CMAKE_CURRENT_LIST_DIR}/config_linux.cmake")

# SIMD for aarch64
set(FLAGS_COMMON ${FLAGS_COMMON}
    "-ftree-vectorize"
    )
