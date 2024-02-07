# A workaround for accessing to NDK_DIR. This works since Env Vars are always accessible
# while cache variables are not
if (NDK_DIR)
    set(ENV{_NDK_DIR} "${NDK_DIR}")
else (NDK_DIR)
    set(NDK_DIR "$ENV{_NDK_DIR}")
endif (NDK_DIR)

if(NOT DEFINED NDK_DIR)
  message(FATAL_ERROR "NDK_DIR should be specified via cmake argument")
endif(NOT DEFINED NDK_DIR)

set(ANDROID_ABI arm64-v8a)
set(ANDROID_API_LEVEL 29)
set(ANDROID_PLATFORM android-${ANDROID_API_LEVEL})
set(ANDROID_STL c++_shared)
set(ANDROID_STL_LIB "${NDK_DIR}/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/libc++_shared.so")

# Find package in the host. `nnfw_find_package` won't work without this
# Others (library, path) will follow android.toolchain.cmake settings
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE NEVER)

# Use the toolchain file that NDK provides
include(${NDK_DIR}/build/cmake/android.toolchain.cmake)

# Install libc++_shared.so to lib folder
install(FILES ${ANDROID_STL_LIB} DESTINATION lib)

# ndk always enable debug flag -g, but we don't want debug info in release build
# https://github.com/android/ndk/issues/243
string(REPLACE "-g" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REPLACE "-g" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "-g ${CMAKE_C_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_DEBUG "-g ${CMAKE_CXX_FLAGS_DEBUG}")

set(TARGET_OS "android")
set(TARGET_ARCH "aarch64")
