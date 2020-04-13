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

# Find package in the host. `nnfw_find_package` won't work without this
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE NEVER)
# Find library in the host. Necessary for `add_library` searching in `out/lib` dir.
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)

# Use the toolchain file that NDK provides
include(${NDK_DIR}/build/cmake/android.toolchain.cmake)

set(TARGET_OS "android")
set(TARGET_ARCH "aarch64")
