# This toolchain file is used to help setting variables for Android cross compilation.
# It is a wrapper of android.toolchain.cmake provided by NDK.
# This file requires only CMAKE_ANDROID_NDK or ANDROID_NDK environment variable.

# If you want to use android NDK's toolchain file directly without this file,
# please follow CMake's android cross build guide
# https://cmake.org/cmake/help/v3.18/manual/cmake-toolchains.7.html#cross-compiling-for-android
# https://developer.android.com/studio/projects/configure-cmake#call-cmake-cli

# Use android sdk environment variable ANDROID_NDK to find the toolchain
# Otherwise, use cmake android cross build guide's CMAKE_ANDROID_NDK variable
if (DEFINED ENV{ANDROID_NDK})
  set(CMAKE_ANDROID_NDK $ENV{ANDROID_NDK})
elseif (CMAKE_ANDROID_NDK)
  # A workaround for accessing NDK path
  # This helps to work on try_compile since Env Vars are always accessible
  # while cache variables are not
  set(ENV{ANDROID_NDK} ${CMAKE_ANDROID_NDK})
endif ()

if (NOT EXISTS ${CMAKE_ANDROID_NDK})
  message(FATAL_ERROR "CMAKE_ANDROID_NDK does not exist: ${CMAKE_ANDROID_NDK}")
endif (NOT EXISTS ${CMAKE_ANDROID_NDK})

set(ANDROID_ABI arm64-v8a)
set(ANDROID_API_LEVEL 29)
set(ANDROID_PLATFORM android-${ANDROID_API_LEVEL})
set(ANDROID_STL c++_shared)

# Use the toolchain file that NDK provides
include(${CMAKE_ANDROID_NDK}/build/cmake/android.toolchain.cmake)

find_library(ANDROID_STL_LIB c++_shared PATHS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE})

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
